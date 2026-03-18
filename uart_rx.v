`timescale 1ns / 1ps
//=============================================================================
// uart_rx.v
// UART Receiver — 8N1 format (8 data bits, No parity, 1 stop bit)
// Baud rate : 115200
// Clock     : 100 MHz  →  clocks per bit = 100_000_000 / 115_200 = 868
//
// Protocol (8N1):
//   IDLE  ──┐
//            └─ START (1 bit, low)
//               DATA  (8 bits, LSB first)
//               STOP  (1 bit, high)
//   → rx_done pulses HIGH for exactly 1 clock cycle when a byte is ready
//   → rx_data holds the received byte until the next byte arrives
//=============================================================================
module uart_rx (
    input  wire       clk,        // 100 MHz system clock
    input  wire       rst,        // Active-high synchronous reset
    input  wire       rx,         // UART serial input (idle = HIGH)
    output reg  [7:0] rx_data,    // Received byte (valid when rx_done=1)
    output reg        rx_done     // Pulses HIGH for 1 cycle when byte ready
);

// ── Baud rate parameters ──────────────────────────────────────────────────────
localparam CLK_FREQ  = 100_000_000;
localparam BAUD_RATE = 115_200;
localparam CLKS_PER_BIT  = CLK_FREQ / BAUD_RATE;          // 868
localparam HALF_BIT      = CLKS_PER_BIT / 2;              // 434  (sample midpoint)

// ── State encoding ────────────────────────────────────────────────────────────
localparam S_IDLE  = 2'd0;
localparam S_START = 2'd1;
localparam S_DATA  = 2'd2;
localparam S_STOP  = 2'd3;

// ── Synchronise async rx line into clock domain (2-FF metastability filter) ──
reg rx_sync0, rx_sync1;
always @(posedge clk) begin
    rx_sync0 <= rx;
    rx_sync1 <= rx_sync0;
end
wire rx_s = rx_sync1;   // synchronised rx

// ── Internal registers ────────────────────────────────────────────────────────
reg [1:0]  state;
reg [9:0]  clk_cnt;     // baud clock counter  (needs ≥10 bits for 868)
reg [2:0]  bit_idx;     // which data bit we are receiving (0..7)
reg [7:0]  rx_shift;    // shift register

// ── State machine ─────────────────────────────────────────────────────────────
always @(posedge clk) begin
    if (rst) begin
        state   <= S_IDLE;
        clk_cnt <= 0;
        bit_idx <= 0;
        rx_shift <= 8'h00;
        rx_data  <= 8'h00;
        rx_done  <= 1'b0;
    end else begin
        rx_done <= 1'b0;          // default: not done

        case (state)

            // ── Wait for falling edge (start bit) ──────────────────────────
            S_IDLE: begin
                if (rx_s == 1'b0) begin   // start bit detected
                    state   <= S_START;
                    clk_cnt <= 0;
                end
            end

            // ── Wait to the MIDDLE of the start bit, then verify it ─────────
            S_START: begin
                if (clk_cnt == HALF_BIT - 1) begin
                    if (rx_s == 1'b0) begin   // still low → valid start bit
                        state   <= S_DATA;
                        clk_cnt <= 0;
                        bit_idx <= 0;
                    end else begin            // glitch — go back to idle
                        state   <= S_IDLE;
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end

            // ── Sample each data bit at the MIDDLE of its bit period ────────
            S_DATA: begin
                if (clk_cnt == CLKS_PER_BIT - 1) begin
                    clk_cnt              <= 0;
                    rx_shift[bit_idx]    <= rx_s;   // LSB first
                    if (bit_idx == 3'd7) begin
                        state   <= S_STOP;
                        bit_idx <= 0;
                    end else begin
                        bit_idx <= bit_idx + 1;
                    end
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end

            // ── Wait for stop bit, then output the byte ─────────────────────
            S_STOP: begin
                if (clk_cnt == CLKS_PER_BIT - 1) begin
                    rx_done  <= 1'b1;         // pulse valid for 1 cycle
                    rx_data  <= rx_shift;
                    state    <= S_IDLE;
                    clk_cnt  <= 0;
                end else begin
                    clk_cnt <= clk_cnt + 1;
                end
            end

            default: state <= S_IDLE;

        endcase
    end
end

endmodule
