`timescale 1ns / 1ps
//=============================================================================
// threat_alert.v
// RGB LED Threat Alert Controller
// Background → Green LED (steady)
// Gunshot    → Red LED (fast blink ~4 Hz)
// Drone      → Yellow LED (slow blink ~1 Hz)
// Designed for Basys 3 / Nexys A7 with individual R, G, B LED pins
//=============================================================================
module threat_alert (
    input  wire        clk,          // System clock (100 MHz typical)
    input  wire        rst,
    input  wire [1:0]  class_in,     // 00=Background, 01=Gunshot, 10=Drone
    input  wire        valid,        // Update when valid
    output reg         led_r,        // Red LED
    output reg         led_g,        // Green LED
    output reg         led_b         // Blue LED (unused, always off)
);

// =============================================================================
// CLOCK DIVIDERS
// For 100 MHz clock:
//   Fast blink (~4 Hz): toggle every 12_500_000 cycles
//   Slow blink (~1 Hz): toggle every 50_000_000 cycles
// =============================================================================
reg [25:0] blink_counter;
wire       fast_blink;  // ~4 Hz toggle
wire       slow_blink;  // ~1 Hz toggle

always @(posedge clk or posedge rst) begin
    if (rst)
        blink_counter <= 26'd0;
    else
        blink_counter <= blink_counter + 1;
end

// For 100 MHz: bit 23 toggles at ~5.96 Hz (close to 4 Hz)
//              bit 25 toggles at ~1.49 Hz (close to 1 Hz)
assign fast_blink = blink_counter[23];
assign slow_blink = blink_counter[25];

// =============================================================================
// LATCH CLASS
// =============================================================================
reg [1:0] latched_class;
always @(posedge clk or posedge rst) begin
    if (rst)
        latched_class <= 2'b00;
    else if (valid)
        latched_class <= class_in;
end

// =============================================================================
// LED OUTPUT LOGIC
// =============================================================================
always @(*) begin
    // Default: all off
    led_r = 1'b0;
    led_g = 1'b0;
    led_b = 1'b0;

    case (latched_class)
        2'b00: begin
            // Background → Green steady
            led_g = 1'b1;
        end
        2'b01: begin
            // Gunshot → Red fast blink
            led_r = fast_blink;
        end
        2'b10: begin
            // Drone → Yellow (Red + Green) slow blink
            led_r = slow_blink;
            led_g = slow_blink;
        end
        default: begin
            led_r = 1'b0;
            led_g = 1'b0;
        end
    endcase
end

endmodule
