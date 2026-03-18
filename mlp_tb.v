`timescale 1ns / 1ps
//=============================================================================
// mlp_tb.v  —  Testbench for mlp_classifier
// Tests 3 representative audio samples (one per class)
// Update the x0/x1/x2/x3 values using output of get_test_vectors.py
//=============================================================================
module mlp_tb;

    // ── DUT signals ──────────────────────────────────────────────────────────
    reg         clk;
    reg         rst;
    reg         valid_in;
    reg  signed [15:0] x0, x1, x2, x3;
    wire [1:0]  class_out;
    wire        valid_out;

    // ── Instantiate DUT ───────────────────────────────────────────────────────
    mlp_classifier uut (
        .clk(clk), .rst(rst), .valid_in(valid_in),
        .x0(x0), .x1(x1), .x2(x2), .x3(x3),
        .class_out(class_out), .valid_out(valid_out)
    );

    // ── Clock: 10ns period (100 MHz) ─────────────────────────────────────────
    always #5 clk = ~clk;

    // ── Human-readable class label ────────────────────────────────────────────
    function [95:0] label;
        input [1:0] cls;
        case (cls)
            2'b00: label = "Background";
            2'b01: label = "GUNSHOT   ";
            2'b10: label = "DRONE     ";
            default: label = "Unknown   ";
        endcase
    endfunction

    // ── Task: apply one sample and wait for result ────────────────────────────
    task apply_sample;
        input signed [15:0] sx0, sx1, sx2, sx3;
        input [95:0] expected_label;
        begin
            x0 = sx0; x1 = sx1; x2 = sx2; x3 = sx3;
            valid_in = 1;
            @(posedge clk); #1;   // inference runs (blocking = single cycle)
            @(posedge clk); #1;   // output registered, valid_out=1
            valid_in = 0;
            #1;
            $display("  Inputs: x0=%4d  x1=%4d  x2=%4d  x3=%4d", sx0, sx1, sx2, sx3);
            $display("  Output: class=%b  label=%-10s  valid=%b", class_out, label(class_out), valid_out);
            $display("  Expected: %-10s  =>  %s",
                expected_label,
                (label(class_out) == expected_label) ? "PASS" : "FAIL");
            $display("");
        end
    endtask

    integer i;

    initial begin
        // ── Initialise ───────────────────────────────────────────────────────
        clk      = 0;
        rst      = 1;
        valid_in = 0;
        x0 = 0; x1 = 0; x2 = 0; x3 = 0;

        // Release reset after 2 cycles
        repeat(2) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);

        $display("=====================================================");
        $display(" MLP Acoustic Threat Classifier — Simulation Results");
        $display("=====================================================");
        $display("");

        // ── TEST 1: Background sample ─────────────────────────────────────────
        // Class 0 — Background (hardware-verified)
        $display("TEST 1 — Expected: Background (class 00)");
        apply_sample(-16'sd670, -16'sd479, 16'sd289, -16'sd355, "Background");

        // ── TEST 2: Gunshot sample ────────────────────────────────────────────
        // Class 1 — Gunshot (hardware-verified)
        $display("TEST 2 — Expected: GUNSHOT (class 01)");
        apply_sample(16'sd154, -16'sd158, 16'sd161, -16'sd373, "GUNSHOT   ");

        // ── TEST 3: Drone sample ──────────────────────────────────────────────
        // Class 2 — Drone (hardware-verified)
        $display("TEST 3 — Expected: DRONE (class 10)");
        apply_sample(16'sd197, -16'sd53, 16'sd34, 16'sd352, "DRONE     ");

        $display("=====================================================");
        $display(" Simulation complete.");
        $display("=====================================================");

        #50;
        $finish;
    end

endmodule
