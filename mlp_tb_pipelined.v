`timescale 1ns / 1ps
//=============================================================================
// mlp_tb_pipelined.v — Testbench for mlp_classifier_pipelined
// Architecture : 16 inputs → 32 hidden (ReLU) → 16 hidden (ReLU) → 3 outputs
// Pipeline     : 4-stage (L1 MAC | ReLU+L2 MAC | ReLU+L3 MAC | Argmax)
// Latency      : 4 clock cycles
// Tests        : 3 representative samples + back-to-back throughput test
//=============================================================================
module mlp_tb_pipelined;

    // ── DUT signals ───────────────────────────────────────────────────────────
    reg        clk;
    reg        rst;
    reg        valid_in;

    // All 16 feature inputs (Q8 signed)
    reg signed [15:0] x0,  x1,  x2,  x3;
    reg signed [15:0] x4,  x5,  x6,  x7;
    reg signed [15:0] x8,  x9,  x10, x11;
    reg signed [15:0] x12, x13, x14, x15;

    wire [1:0]         class_out;
    wire               valid_out;
    wire signed [31:0] conf_delta;
    wire               threat;

    // ── Instantiate DUT ───────────────────────────────────────────────────────
    mlp_classifier_pipelined uut (
        .clk      (clk),
        .rst      (rst),
        .valid_in (valid_in),
        // 16 feature inputs
        .x0 (x0),  .x1 (x1),  .x2 (x2),  .x3 (x3),
        .x4 (x4),  .x5 (x5),  .x6 (x6),  .x7 (x7),
        .x8 (x8),  .x9 (x9),  .x10(x10), .x11(x11),
        .x12(x12), .x13(x13), .x14(x14), .x15(x15),
        // outputs
        .class_out (class_out),
        .valid_out (valid_out),
        .conf_delta(conf_delta),
        .threat    (threat)
    );

    // ── Clock: 10 ns period (100 MHz) ─────────────────────────────────────────
    always #5 clk = ~clk;

    // ── Waveform dump ─────────────────────────────────────────────────────────
    initial begin
        $dumpfile("mlp_pipelined.vcd");
        $dumpvars(0, mlp_tb_pipelined);
    end

    // ── Human-readable class label ────────────────────────────────────────────
    function [95:0] label;
        input [1:0] cls;
        case (cls)
            2'b00:   label = "Background";
            2'b01:   label = "GUNSHOT   ";
            2'b10:   label = "DRONE     ";
            default: label = "Unknown   ";
        endcase
    endfunction

    // ── Helper task: zero all inputs ──────────────────────────────────────────
    task zero_inputs;
        begin
            x0=0; x1=0; x2=0;  x3=0;
            x4=0; x5=0; x6=0;  x7=0;
            x8=0; x9=0; x10=0; x11=0;
            x12=0; x13=0; x14=0; x15=0;
        end
    endtask

    // ── Task: apply one 16-feature sample, wait 4 cycles, check output ────────
    integer pass_count;
    integer fail_count;

    task apply_and_check;
        // 16 feature inputs
        input signed [15:0] sx0,  sx1,  sx2,  sx3;
        input signed [15:0] sx4,  sx5,  sx6,  sx7;
        input signed [15:0] sx8,  sx9,  sx10, sx11;
        input signed [15:0] sx12, sx13, sx14, sx15;
        // expected result
        input [1:0]   expected_class;
        input [95:0]  expected_label;
        input integer test_num;
        begin
            // Drive all 16 inputs
            x0=sx0;   x1=sx1;   x2=sx2;   x3=sx3;
            x4=sx4;   x5=sx5;   x6=sx6;   x7=sx7;
            x8=sx8;   x9=sx9;   x10=sx10; x11=sx11;
            x12=sx12; x13=sx13; x14=sx14; x15=sx15;
            valid_in = 1;
            @(posedge clk); #1;
            valid_in = 0;
            zero_inputs;

            // Wait for 4-stage pipeline to produce result (5 edges total)
            @(posedge clk); #1;   // stage 1 → stage 2
            @(posedge clk); #1;   // stage 2 → stage 3
            @(posedge clk); #1;   // stage 3 → stage 4
            @(posedge clk); #1;   // stage 4 output registered
            @(posedge clk); #1;   // output stable — read here

            $display("  TEST %0d", test_num);
            $display("    Inputs (x0-x3): %4d  %4d  %4d  %4d", sx0, sx1, sx2, sx3);
            $display("    Inputs (x4-x7): %4d  %4d  %4d  %4d", sx4, sx5, sx6, sx7);
            $display("    Inputs (x8-x11):%4d  %4d  %4d  %4d", sx8, sx9, sx10, sx11);
            $display("    Inputs(x12-x15):%4d  %4d  %4d  %4d", sx12, sx13, sx14, sx15);
            $display("    valid_out = %b", valid_out);
            $display("    class_out = %b  (%s)  threat=%b", class_out, label(class_out), threat);
            $display("    conf_delta= %0d", conf_delta);

            if (class_out == expected_class) begin
                $display("    Expected: %-10s => PASS", expected_label);
                pass_count = pass_count + 1;
            end else begin
                $display("    Expected: %-10s => FAIL (got %s)", expected_label, label(class_out));
                fail_count = fail_count + 1;
            end
            $display("");
        end
    endtask

    // ── Main test sequence ────────────────────────────────────────────────────
    initial begin
        clk      = 0;
        rst      = 1;
        valid_in = 0;
        zero_inputs;
        pass_count = 0;
        fail_count = 0;

        // Release reset after 2 cycles
        repeat(2) @(posedge clk);
        rst = 0;
        repeat(2) @(posedge clk);

        $display("=================================================================");
        $display(" MLP Pipelined Classifier — Simulation Results");
        $display(" Architecture: 16->32->16->3  |  4-stage pipeline  |  latency=4");
        $display("=================================================================");
        $display("");

        // ── TEST 1: Background ────────────────────────────────────────────────
        // Features: MFCCs x0-x9, then spectral/band features x10-x15
        // NOTE: With all-zero weights these will all output class=00 (Background).
        //       Replace test vectors with get_test_vectors.py output after training.
        apply_and_check(
            -16'sd673, -16'sd481,  16'sd291, -16'sd354,   // x0-x3
             16'sd278, -16'sd329,  16'sd254, -16'sd265,   // x4-x7
             16'sd157, -16'sd136, -16'sd471, -16'sd495,   // x8-x11
            -16'sd327,  -16'sd85,  -16'sd78,  -16'sd65,  // x12-x15
            2'b00, "Background", 1
        );

        // ── TEST 2: Gunshot ───────────────────────────────────────────────────
        apply_and_check(
             16'sd153, -16'sd159,  16'sd162, -16'sd372,   // x0-x3
             16'sd227, -16'sd394,  16'sd140, -16'sd269,   // x4-x7
            -16'sd31,  -16'sd228,  16'sd262,  16'sd345,   // x8-x11
             16'sd118,    16'sd8,   16'sd73,   16'sd35,   // x12-x15
            2'b01, "GUNSHOT   ", 2
        );

        // ── TEST 3: Drone ─────────────────────────────────────────────────────
        apply_and_check(
             16'sd197,  -16'sd54,   16'sd35,  16'sd352,   // x0-x3
              16'sd63,  16'sd248,   16'sd51,  16'sd198,   // x4-x7
              16'sd82,  16'sd241,  16'sd126,  16'sd191,   // x8-x11
              16'sd38,   16'sd89,  -16'sd66,  -16'sd35,   // x12-x15
            2'b10, "DRONE     ", 3
        );

        // ── TEST 4: Low-confidence Gunshot (should suppress threat) ───────────
        // We use a modified version of the gunshot sample with all values 
        // shifted down significantly to reduce confidence relative to biases.
        apply_and_check(
             16'sd15, -16'sd16,  16'sd16, -16'sd37,   // x0-x3 (scaled down by 10)
             16'sd23, -16'sd39,  16'sd14, -16'sd27,   // x4-x7
            -16'sd3,  -16'sd23,  16'sd26,  16'sd35,   // x8-x11
             16'sd12,    16'sd1,   16'sd7,   16'sd4,   // x12-x15
            2'b01, "GUNSHOT (L)", 4
        );

        $display("=================================================================");
        $display(" PIPELINE THROUGHPUT TEST — 3 samples fed in consecutive cycles");
        $display("=================================================================");
        $display("");

        // Sample 1 (Background) — cycle N
        x0=-16'sd673; x1=-16'sd481; x2= 16'sd291; x3=-16'sd354;
        x4= 16'sd278; x5=-16'sd329; x6= 16'sd254; x7=-16'sd265;
        x8= 16'sd157; x9=-16'sd136; x10=-16'sd471; x11=-16'sd495;
        x12=-16'sd327; x13=-16'sd85; x14=-16'sd78; x15=-16'sd65;
        valid_in = 1;
        @(posedge clk); #1;

        // Sample 2 (Gunshot) — cycle N+1
        x0= 16'sd153; x1=-16'sd159; x2= 16'sd162; x3=-16'sd372;
        x4= 16'sd227; x5=-16'sd394; x6= 16'sd140; x7=-16'sd269;
        x8=-16'sd31;  x9=-16'sd228; x10=16'sd262; x11= 16'sd345;
        x12=16'sd118; x13=  16'sd8; x14= 16'sd73; x15=  16'sd35;
        @(posedge clk); #1;

        // Sample 3 (Drone) — cycle N+2
        x0= 16'sd197; x1= -16'sd54; x2=  16'sd35; x3= 16'sd352;
        x4=  16'sd63; x5= 16'sd248; x6=  16'sd51; x7= 16'sd198;
        x8=  16'sd82; x9= 16'sd241; x10=16'sd126; x11=16'sd191;
        x12= 16'sd38; x13= 16'sd89; x14=-16'sd66; x15=-16'sd35;
        @(posedge clk); #1;

        valid_in = 0;
        zero_inputs;

        // Results emerge 5 cycles after each input
        // Output 1 ready at cycle N+5
        @(posedge clk); #1;
        @(posedge clk); #1;
        $display("  Pipeline output 1 (expect Background): class=%b (%s) threat=%b conf=%0d",
                 class_out, label(class_out), threat, conf_delta);

        // Output 2 ready at cycle N+5
        @(posedge clk); #1;
        $display("  Pipeline output 2 (expect Gunshot):    class=%b (%s) threat=%b conf=%0d",
                 class_out, label(class_out), threat, conf_delta);

        // Output 3 ready at cycle N+6
        @(posedge clk); #1;
        $display("  Pipeline output 3 (expect Drone):      class=%b (%s) threat=%b conf=%0d",
                 class_out, label(class_out), threat, conf_delta);

        $display("");
        $display("=================================================================");
        $display(" RESULTS: %0d PASS  /  %0d FAIL", pass_count, fail_count);
        $display(" NOTE: With placeholder (zero) weights all tests will output");
        $display("       class=00 (Background). Replace test vectors using");
        $display("       get_test_vectors.py after training mlp_model.pkl.");
        $display("=================================================================");

        #50;
        $finish;
    end

endmodule