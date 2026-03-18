`timescale 1ns / 1ps
//=============================================================================
// seven_seg_display.v
// 4-digit multiplexed 7-segment display controller
// Displays class label:  bg (Background) | gun (Gunshot) | drn (Drone)
// Compatible with Basys 3 / Nexys A7 (common-anode, active-low segments)
//=============================================================================
module seven_seg_display (
    input  wire        clk,          // System clock (100 MHz typical)
    input  wire        rst,
    input  wire [1:0]  class_in,     // 00=Background, 01=Gunshot, 10=Drone
    input  wire        valid,        // Display updates only when valid
    output reg  [3:0]  an,           // Anode enable (active-low, 4 digits)
    output reg  [6:0]  seg           // Segments a-g (active-low)
);

// =============================================================================
// REFRESH COUNTER — cycles through 4 digits at ~1 kHz
// For 100 MHz clock: divide by 100_000 → ~1 kHz refresh
// For simulation: uses fewer bits — still cycles correctly
// =============================================================================
reg [16:0] refresh_counter;
wire [1:0] digit_select;

always @(posedge clk or posedge rst) begin
    if (rst)
        refresh_counter <= 17'd0;
    else
        refresh_counter <= refresh_counter + 1;
end

assign digit_select = refresh_counter[16:15]; // 2 MSBs select digit

// =============================================================================
// CHARACTER STORAGE — what to display per digit
// =============================================================================
// Encoding: custom characters mapped to 7-seg patterns
// We store 4 characters per class label
reg [4:0] char0, char1, char2, char3; // 5-bit char code per digit

// Character codes:
localparam C_BLANK = 5'd0;
localparam C_B     = 5'd1;   // b
localparam C_G     = 5'd2;   // g  (or G)
localparam C_U     = 5'd3;   // U
localparam C_N     = 5'd4;   // n
localparam C_D     = 5'd5;   // d
localparam C_R     = 5'd6;   // r
localparam C_O     = 5'd7;   // o (same as degree symbol)
localparam C_E     = 5'd8;   // E (for "nE" — drone)

// Latch the class label when valid
reg [1:0] latched_class;
always @(posedge clk or posedge rst) begin
    if (rst)
        latched_class <= 2'b00;
    else if (valid)
        latched_class <= class_in;
end

// Map class → 4 characters (right-justified, leftmost is digit3)
//   Background → " b g " → [BLANK, B, G, BLANK]  → shows "bG"
//   Gunshot    → "G U n"  → [G, U, N, BLANK]       → shows "GUn"
//   Drone      → "d r n"  → [D, R, N, BLANK]       → shows "drn"
always @(*) begin
    case (latched_class)
        2'b00: begin char3=C_BLANK; char2=C_BLANK; char1=C_B;     char0=C_G;     end // " bG"
        2'b01: begin char3=C_BLANK; char2=C_G;     char1=C_U;     char0=C_N;     end // "GUn"
        2'b10: begin char3=C_BLANK; char2=C_D;     char1=C_R;     char0=C_N;     end // "drn"
        default: begin char3=C_BLANK; char2=C_BLANK; char1=C_BLANK; char0=C_BLANK; end
    endcase
end

// =============================================================================
// 7-SEGMENT DECODER — character code → segment pattern (active-low)
//
//   Segment mapping:     a
//                       ---
//                    f |   | b
//                       -g-
//                    e |   | c
//                       ---
//                        d
//   seg[6:0] = {a, b, c, d, e, f, g}   (active-low)
// =============================================================================
reg [4:0] current_char;

always @(*) begin
    case (digit_select)
        2'b00: begin an = 4'b1110; current_char = char0; end  // rightmost digit
        2'b01: begin an = 4'b1101; current_char = char1; end
        2'b10: begin an = 4'b1011; current_char = char2; end
        2'b11: begin an = 4'b0111; current_char = char3; end
    endcase
end

always @(*) begin
    case (current_char)
        //                         abcdefg
        C_BLANK: seg = 7'b1111111; // all off
        C_B:     seg = 7'b1100000; // b: segments c,d,e,f,g
        C_G:     seg = 7'b0100001; // G: a,c,d,e,f
        C_U:     seg = 7'b1000001; // U: b,c,d,e,f
        C_N:     seg = 7'b1101010; // n: c,e,g
        C_D:     seg = 7'b1000010; // d: b,c,d,e,g
        C_R:     seg = 7'b1111010; // r: e,g
        C_O:     seg = 7'b1100010; // o: c,d,e,g
        C_E:     seg = 7'b0100001; // E: a,d,e,f,g → same as used for E
        default: seg = 7'b1111111;
    endcase
end

endmodule
