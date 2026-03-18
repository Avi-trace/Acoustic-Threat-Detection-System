`timescale 1ns / 1ps
//=============================================================================
// mlp_classifier.v
// Multi-Threat Acoustic Classifier — MLP Inference Engine
// Architecture: 4 inputs → 8 hidden neurons (ReLU) → 3 outputs (argmax)
// Fixed-point: Q8 format (weights/inputs scaled by 256)
// Classes: 00 = Background | 01 = Gunshot | 10 = Drone
//=============================================================================
module mlp_classifier (
    input  wire        clk,
    input  wire        rst,
    input  wire        valid_in,
    input  wire signed [15:0] x0,   // MFCC feature 0 (Q8 fixed-point)
    input  wire signed [15:0] x1,   // MFCC feature 1
    input  wire signed [15:0] x2,   // MFCC feature 2
    input  wire signed [15:0] x3,   // MFCC feature 3
    output reg  [1:0]  class_out,   // 00=Background 01=Gunshot 10=Drone
    output reg         valid_out
);

// =============================================================================
// LAYER 1 WEIGHTS — W1[input_idx][neuron_idx] — shape (4,8) — Q8
// =============================================================================
// Neuron 0
localparam signed [15:0] W1_0_0 = -5,    W1_1_0 = 12,   W1_2_0 = -43,  W1_3_0 = -4;
// Neuron 1
localparam signed [15:0] W1_0_1 = 52,    W1_1_1 = 336,  W1_2_1 = -349, W1_3_1 = -132;
// Neuron 2
localparam signed [15:0] W1_0_2 = 228,   W1_1_2 = -118, W1_2_2 = 155,  W1_3_2 = -35;
// Neuron 3
localparam signed [15:0] W1_0_3 = 27,    W1_1_3 = 85,   W1_2_3 = 105,  W1_3_3 = 134;
// Neuron 4
localparam signed [15:0] W1_0_4 = -443,  W1_1_4 = 125,  W1_2_4 = 220,  W1_3_4 = 123;
// Neuron 5
localparam signed [15:0] W1_0_5 = -299,  W1_1_5 = -430, W1_2_5 = -121, W1_3_5 = -186;
// Neuron 6
localparam signed [15:0] W1_0_6 = -3,    W1_1_6 = -74,  W1_2_6 = 67,   W1_3_6 = 386;
// Neuron 7
localparam signed [15:0] W1_0_7 = 669,   W1_1_7 = -213, W1_2_7 = 197,  W1_3_7 = -204;

// =============================================================================
// LAYER 1 BIASES — b1[neuron_idx] — Q8
// =============================================================================
localparam signed [15:0] B1_0 = -171, B1_1 = 305,  B1_2 = 252,  B1_3 = 68;
localparam signed [15:0] B1_4 = 205,  B1_5 = -77,  B1_6 = 224,  B1_7 = -176;

// =============================================================================
// LAYER 2 WEIGHTS — W2[neuron_idx][output_idx] — shape (8,3) — Q8
// =============================================================================
// From neuron 0 → outputs [Background, Gunshot, Drone]
localparam signed [15:0] W2_0_0 = -23,  W2_0_1 = -1,   W2_0_2 = 0;
// From neuron 1
localparam signed [15:0] W2_1_0 = 351,  W2_1_1 = -198, W2_1_2 = -12;
// From neuron 2
localparam signed [15:0] W2_2_0 = -130, W2_2_1 = 76,   W2_2_2 = 71;
// From neuron 3
localparam signed [15:0] W2_3_0 = -53,  W2_3_1 = -44,  W2_3_2 = 139;
// From neuron 4
localparam signed [15:0] W2_4_0 = 449,  W2_4_1 = -142, W2_4_2 = -188;
// From neuron 5
localparam signed [15:0] W2_5_0 = 397,  W2_5_1 = -398, W2_5_2 = -320;
// From neuron 6
localparam signed [15:0] W2_6_0 = -72,  W2_6_1 = -857, W2_6_2 = 109;
// From neuron 7
localparam signed [15:0] W2_7_0 = 40,   W2_7_1 = 526,  W2_7_2 = -884;

// =============================================================================
// LAYER 2 BIASES — b2[output_idx] — Q8
// =============================================================================
localparam signed [15:0] B2_0 = -18, B2_1 = -109, B2_2 = -141;

// =============================================================================
// INTERNAL SIGNALS
// =============================================================================
reg signed [31:0] net1 [0:7];   // Layer 1 pre-activation (MAC result)
reg signed [31:0] h    [0:7];   // Layer 1 post-ReLU
reg signed [31:0] out  [0:2];   // Layer 2 output scores

// =============================================================================
// INFERENCE PIPELINE (single-cycle, clocked)
// =============================================================================
always @(posedge clk or posedge rst) begin
    if (rst) begin
        class_out <= 2'b00;
        valid_out <= 1'b0;

    end else if (valid_in) begin

        // ── LAYER 1: Multiply-Accumulate (blocking = sequential within one cycle) ─
        net1[0] = (($signed(x0)*W1_0_0 + $signed(x1)*W1_1_0 + $signed(x2)*W1_2_0 + $signed(x3)*W1_3_0) >>> 8) + B1_0;
        net1[1] = (($signed(x0)*W1_0_1 + $signed(x1)*W1_1_1 + $signed(x2)*W1_2_1 + $signed(x3)*W1_3_1) >>> 8) + B1_1;
        net1[2] = (($signed(x0)*W1_0_2 + $signed(x1)*W1_1_2 + $signed(x2)*W1_2_2 + $signed(x3)*W1_3_2) >>> 8) + B1_2;
        net1[3] = (($signed(x0)*W1_0_3 + $signed(x1)*W1_1_3 + $signed(x2)*W1_2_3 + $signed(x3)*W1_3_3) >>> 8) + B1_3;
        net1[4] = (($signed(x0)*W1_0_4 + $signed(x1)*W1_1_4 + $signed(x2)*W1_2_4 + $signed(x3)*W1_3_4) >>> 8) + B1_4;
        net1[5] = (($signed(x0)*W1_0_5 + $signed(x1)*W1_1_5 + $signed(x2)*W1_2_5 + $signed(x3)*W1_3_5) >>> 8) + B1_5;
        net1[6] = (($signed(x0)*W1_0_6 + $signed(x1)*W1_1_6 + $signed(x2)*W1_2_6 + $signed(x3)*W1_3_6) >>> 8) + B1_6;
        net1[7] = (($signed(x0)*W1_0_7 + $signed(x1)*W1_1_7 + $signed(x2)*W1_2_7 + $signed(x3)*W1_3_7) >>> 8) + B1_7;

        // ── ReLU Activation: f(x) = max(0, x) ───────────────────────────────
        h[0] = (net1[0] > 0) ? net1[0] : 32'sd0;
        h[1] = (net1[1] > 0) ? net1[1] : 32'sd0;
        h[2] = (net1[2] > 0) ? net1[2] : 32'sd0;
        h[3] = (net1[3] > 0) ? net1[3] : 32'sd0;
        h[4] = (net1[4] > 0) ? net1[4] : 32'sd0;
        h[5] = (net1[5] > 0) ? net1[5] : 32'sd0;
        h[6] = (net1[6] > 0) ? net1[6] : 32'sd0;
        h[7] = (net1[7] > 0) ? net1[7] : 32'sd0;

        // ── LAYER 2: Multiply-Accumulate ─────────────────────────────────────
        out[0] = ((h[0]*W2_0_0 + h[1]*W2_1_0 + h[2]*W2_2_0 + h[3]*W2_3_0 +
                   h[4]*W2_4_0 + h[5]*W2_5_0 + h[6]*W2_6_0 + h[7]*W2_7_0) >>> 8) + B2_0;
        out[1] = ((h[0]*W2_0_1 + h[1]*W2_1_1 + h[2]*W2_2_1 + h[3]*W2_3_1 +
                   h[4]*W2_4_1 + h[5]*W2_5_1 + h[6]*W2_6_1 + h[7]*W2_7_1) >>> 8) + B2_1;
        out[2] = ((h[0]*W2_0_2 + h[1]*W2_1_2 + h[2]*W2_2_2 + h[3]*W2_3_2 +
                   h[4]*W2_4_2 + h[5]*W2_5_2 + h[6]*W2_6_2 + h[7]*W2_7_2) >>> 8) + B2_2;

        // ── Argmax: class with highest score wins ────────────────────────────
        if (out[0] >= out[1] && out[0] >= out[2])
            class_out <= 2'b00;   // Background
        else if (out[1] >= out[0] && out[1] >= out[2])
            class_out <= 2'b01;   // Gunshot  ← THREAT
        else
            class_out <= 2'b10;   // Drone    ← THREAT

        valid_out <= 1'b1;

    end else begin
        valid_out <= 1'b0;
    end
end

endmodule
