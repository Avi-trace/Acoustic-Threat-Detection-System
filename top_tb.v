`timescale 1ns / 1ps
//=============================================================================
// top_tb.v — Integration Testbench for top_module
// Verifies classifier + 7-segment display + threat LEDs working together
//=============================================================================
module top_tb;

    // ── DUT signals ──────────────────────────────────────────────────────────
    reg         clk;
    reg         rst;
    reg         btn_classify;
    reg  signed [15:0] x0, x1, x2, x3;
    wire [3:0]  an;
    wire [6:0]  seg;
    wire        led_r, led_g, led_b;
    wire [1:0]  class_out;
    wire        threat;
    wire        valid_out;

    // ── Instantiate DUT ───────────────────────────────────────────────────────
    top_module uut (
        .clk(clk), .rst(rst),
        .btn_classify(btn_classify),
        .x0(x0), .x1(x1), .x2(x2), .x3(x3),
        .an(an), .seg(seg),
        .led_r(led_r), .led_g(led_g), .led_b(led_b),
        .class_out(class_out), .threat(threat),
        .valid_out(valid_out)
    );

    // ── Clock: 10ns period (100 MHz) ─────────────────────────────────────────
    always #5 clk = ~clk;

    // ── Waveform dump ────────────────────────────────────────────────────────
    initial begin
        $dumpfile("top_module.vcd");
        $dumpvars(0, top_tb);
    end

    // ── Human-readable helpers ───────────────────────────────────────────────
    function [95:0] class_label;
        input [1:0] cls;
        case (cls)
            2'b00: class_label = "Background";
            2'b01: class_label = "GUNSHOT   ";
            2'b10: class_label = "DRONE     ";
            default: class_label = "Unknown   ";
        endcase
    endfunction

    // ── Task: simulate button press (3 cycles high = debounce pass) ─────────
    task press_button;
        begin
            btn_classify = 1;
            repeat(3) @(posedge clk);
            #1;
            btn_classify = 0;
        end
    endtask

    // ── Task: classify and report full system state ──────────────────────────
    task classify_and_report;
        input signed [15:0] sx0, sx1, sx2, sx3;
        input [95:0] expected;
        input integer test_num;
        begin
            // Set inputs
            x0 = sx0; x1 = sx1; x2 = sx2; x3 = sx3;

            // Press classify button
            press_button;

            // Wait for pipeline (3 stages) + extra margin
            repeat(6) @(posedge clk);
            #1;

            $display("  TEST %0d", test_num);
            $display("    Class:   %b (%s)", class_out, class_label(class_out));
            $display("    Threat:  %b", threat);
            $display("    LEDs:    R=%b  G=%b  B=%b", led_r, led_g, led_b);
            $display("    7-Seg:   an=%b  seg=%b", an, seg);
            $display("    Valid:   %b", valid_out);
            if (class_label(class_out) == expected)
                $display("    Result:  PASS");
            else
                $display("    Result:  FAIL (expected %s)", expected);
            $display("");

            // Let display multiplex for a few cycles
            repeat(10) @(posedge clk);
        end
    endtask

    initial begin
        // ── Initialise ───────────────────────────────────────────────────────
        clk          = 0;
        rst          = 1;
        btn_classify = 0;
        x0 = 0; x1 = 0; x2 = 0; x3 = 0;

        // Release reset
        repeat(4) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);

        $display("==========================================================");
        $display(" TOP MODULE Integration Test");
        $display(" Classifier + 7-Segment Display + Threat Alert LEDs");
        $display("==========================================================");
        $display("");

        // ── TEST 1: Background ───────────────────────────────────────────────
        $display("--- Classifying: Background ---");
        classify_and_report(-16'sd670, -16'sd479, 16'sd289, -16'sd355, "Background", 1);

        // ── TEST 2: Gunshot ──────────────────────────────────────────────────
        $display("--- Classifying: Gunshot ---");
        classify_and_report(16'sd154, -16'sd158, 16'sd161, -16'sd373, "GUNSHOT   ", 2);

        // ── TEST 3: Drone ────────────────────────────────────────────────────
        $display("--- Classifying: Drone ---");
        classify_and_report(16'sd197, -16'sd53, 16'sd34, 16'sd352, "DRONE     ", 3);

        // ── LED blink observation ────────────────────────────────────────────
        $display("--- Observing LED blink pattern (100 cycles) ---");
        $display("  Current class: %s  threat=%b", class_label(class_out), threat);
        repeat(100) begin
            @(posedge clk); #1;
        end
        $display("  LED state after 100 cycles: R=%b G=%b B=%b", led_r, led_g, led_b);
        $display("");

        $display("==========================================================");
        $display(" Integration test complete.");
        $display("==========================================================");

        #100;
        $finish;
    end

endmodule
