`timescale 1ns / 1ps
//=============================================================================
// top_module.v
// Top-Level FPGA Integration — Multi-Threat Acoustic Classifier
//
// Data flow:
//   PC  ──UART──►  uart_rx  ──►  Feature Buffer State Machine
//                                        │
//                                        ▼  (after 32 bytes = 16 features)
//                              mlp_classifier_pipelined
//                                        │
//                              ┌─────────┼──────────┐
//                              ▼         ▼          ▼
//                        7-seg display  LEDs    class_out / threat
//
// UART protocol (host → FPGA):
//   • Send 32 bytes: feature[0]_HIGH, feature[0]_LOW,
//                    feature[1]_HIGH, feature[1]_LOW, ...
//                    feature[15]_HIGH, feature[15]_LOW
//   • Each feature is a signed 16-bit value in Q8 fixed-point format,
//     transmitted big-endian (MSB byte first).
//   • After all 32 bytes are received the classifier fires automatically.
//   • The FPGA sends no acknowledgement — poll valid_out / threat LEDs.
//
// Baud  : 115200  |  Clock : 100 MHz  |  Format : 8N1
//=============================================================================
module top_module (
    input  wire        clk,           // 100 MHz system clock
    input  wire        rst,           // Active-high reset (btnC on Basys3/Zybo)
    input  wire        uart_rxd,      // UART RX pin (connect to PC TX)

    // ── 7-Segment display (Basys3 / Zybo compatible) ─────────────────────────
    output wire [3:0]  an,            // Anode enable (active-low)
    output wire [6:0]  seg,           // Segments a-g (active-low)

    // ── Threat alert LEDs ────────────────────────────────────────────────────
    output wire        led_r,         // RED   — Gunshot detected
    output wire        led_g,         // GREEN — Background (safe)
    output wire        led_b,         // BLUE  — Drone detected

    // ── Status / debug ports ─────────────────────────────────────────────────
    output wire [1:0]  class_out,     // Raw class code (for logic analyser)
    output wire        threat,        // HIGH when Gunshot or Drone
    output wire        valid_out,     // Pulses HIGH when classification ready

    // ── Majority Vote Buffers (NEW) ──────────────────────────────────────────
    output reg  [1:0]  confirmed_class_out,
    output reg         confirmed_threat
);

//=============================================================================
// UART RECEIVER
//=============================================================================
wire [7:0] uart_byte;
wire       uart_done;

uart_rx #(
    // Parameters match defaults in uart_rx.v (100 MHz / 115200 baud)
) uart_inst (
    .clk     (clk),
    .rst     (rst),
    .rx      (uart_rxd),
    .rx_data (uart_byte),
    .rx_done (uart_done)
);

//=============================================================================
// FEATURE BUFFER — 16 × 16-bit signed values
// Filled by the receive state machine below.
//=============================================================================
reg signed [15:0] feat [0:15];   // feature[0] = x0 … feature[15] = x15

//=============================================================================
// RECEIVE STATE MACHINE
// Collects 32 bytes (2 per feature, big-endian) then fires the classifier.
//
// States:
//   RX_WAIT    — idle, waiting for first byte
//   RX_HI      — received high byte, storing, waiting for low byte
//   RX_LO      — received low byte, feature complete; advance or trigger
//   RX_TRIGGER — pulse valid_in to classifier for 1 clock cycle
//   RX_PAUSE   — wait for classifier pipeline to drain before re-arming
//=============================================================================
localparam RX_WAIT    = 3'd0;
localparam RX_HI      = 3'd1;
localparam RX_LO      = 3'd2;
localparam RX_TRIGGER = 3'd3;
localparam RX_PAUSE   = 3'd4;

reg [2:0]  rx_state;
reg [3:0]  feat_idx;     // which feature (0..15) we are filling
reg [7:0]  hi_byte;      // temporary: stores the high byte
reg        mlp_valid_in; // feeds valid_in on the classifier
reg [2:0]  pause_cnt;    // short pause counter after trigger

always @(posedge clk) begin
    if (rst) begin
        rx_state    <= RX_WAIT;
        feat_idx    <= 4'd0;
        hi_byte     <= 8'h00;
        mlp_valid_in <= 1'b0;
        pause_cnt   <= 3'd0;
        // Clear feature buffer
        feat[0]  <= 16'sd0; feat[1]  <= 16'sd0;
        feat[2]  <= 16'sd0; feat[3]  <= 16'sd0;
        feat[4]  <= 16'sd0; feat[5]  <= 16'sd0;
        feat[6]  <= 16'sd0; feat[7]  <= 16'sd0;
        feat[8]  <= 16'sd0; feat[9]  <= 16'sd0;
        feat[10] <= 16'sd0; feat[11] <= 16'sd0;
        feat[12] <= 16'sd0; feat[13] <= 16'sd0;
        feat[14] <= 16'sd0; feat[15] <= 16'sd0;
    end else begin
        mlp_valid_in <= 1'b0;   // default

        case (rx_state)

            // ── Wait for the high byte of the next feature ──────────────────
            RX_WAIT: begin
                if (uart_done) begin
                    hi_byte  <= uart_byte;
                    rx_state <= RX_LO;
                end
            end

            // ── Receive low byte, assemble signed 16-bit feature ────────────
            RX_LO: begin
                if (uart_done) begin
                    // Big-endian assembly: {hi_byte, low_byte}
                    feat[feat_idx] <= $signed({hi_byte, uart_byte});

                    if (feat_idx == 4'd15) begin
                        // All 16 features received — trigger classifier
                        feat_idx <= 4'd0;
                        rx_state <= RX_TRIGGER;
                    end else begin
                        feat_idx <= feat_idx + 1;
                        rx_state <= RX_WAIT;
                    end
                end
            end

            // ── Pulse valid_in for exactly 1 clock cycle ─────────────────────
            RX_TRIGGER: begin
                mlp_valid_in <= 1'b1;
                pause_cnt    <= 3'd0;
                rx_state     <= RX_PAUSE;
            end

            // ── Wait for pipeline to drain then re-arm ───────────────────────
            // Pipeline latency = 4 cycles (16→32→16→3), add margin → wait 7 cycles
            RX_PAUSE: begin
                if (pause_cnt == 3'd6) begin
                    rx_state <= RX_WAIT;
                end else begin
                    pause_cnt <= pause_cnt + 1;
                end
            end

            default: rx_state <= RX_WAIT;

        endcase
    end
end

//=============================================================================
// MLP CLASSIFIER (Pipelined, 16 inputs → 32 hidden → 16 hidden → 3 outputs)
//=============================================================================
wire [1:0]         cls;
wire               vld;
wire signed [31:0] conf;
wire               thr;

mlp_classifier_pipelined classifier (
    .clk       (clk),
    .rst       (rst),
    .valid_in  (mlp_valid_in),
    // ── Feature inputs from buffer ──────────────────────────────────────────
    .x0  (feat[0]),  .x1  (feat[1]),  .x2  (feat[2]),  .x3  (feat[3]),
    .x4  (feat[4]),  .x5  (feat[5]),  .x6  (feat[6]),  .x7  (feat[7]),
    .x8  (feat[8]),  .x9  (feat[9]),  .x10 (feat[10]), .x11 (feat[11]),
    .x12 (feat[12]), .x13 (feat[13]), .x14 (feat[14]), .x15 (feat[15]),
    // ── Outputs ─────────────────────────────────────────────────────────────
    .class_out (cls),
    .valid_out (vld),
    .conf_delta(conf),
    .threat    (thr)
);

assign class_out = cls;
assign valid_out = vld;
assign threat    = thr;

//=============================================================================
// 3-FRAME MAJORITY VOTE BUFFER
//=============================================================================
reg [1:0] cls_hist [0:2];
reg       thr_hist [0:2];

always @(posedge clk) begin
    if (rst) begin
        cls_hist[0] <= 2'b00; cls_hist[1] <= 2'b00; cls_hist[2] <= 2'b00;
        thr_hist[0] <= 1'b0;  thr_hist[1] <= 1'b0;  thr_hist[2] <= 1'b0;
        confirmed_class_out <= 2'b00;
        confirmed_threat    <= 1'b0;
    end else if (vld) begin
        // Shift register
        cls_hist[2] <= cls_hist[1];
        cls_hist[1] <= cls_hist[0];
        cls_hist[0] <= cls;
        
        thr_hist[2] <= thr_hist[1];
        thr_hist[1] <= thr_hist[0];
        thr_hist[0] <= thr;

        // Majority vote (2 out of 3)
        confirmed_threat <= ( (thr + thr_hist[0] + thr_hist[1]) >= 2 );
        
        // Majority class vote
        if (cls == cls_hist[0] || cls == cls_hist[1])
            confirmed_class_out <= cls;
        else if (cls_hist[0] == cls_hist[1])
            confirmed_class_out <= cls_hist[0];
        else
            confirmed_class_out <= cls; // Fallback to current if no match
    end
end

//=============================================================================
// 7-SEGMENT DISPLAY
//=============================================================================
seven_seg_display display (
    .clk     (clk),
    .rst     (rst),
    .class_in(cls),
    .valid   (vld),
    .an      (an),
    .seg     (seg)
);

//=============================================================================
// THREAT ALERT LEDs
//=============================================================================
threat_alert alert (
    .clk     (clk),
    .rst     (rst),
    .class_in(cls),
    .valid   (vld),
    .led_r   (led_r),
    .led_g   (led_g),
    .led_b   (led_b)
);

endmodule