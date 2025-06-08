module adder (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,         // 乘法开始信号
    input  wire [31:0] multiplicand,  // 乘数1
    input  wire [31:0] multiplier,    // 乘数2

    output reg  [63:0] product,       // 乘法结果
    output wire        done           // 乘法完成信号
);

    reg [63:0] acc;       // 累加器
    reg [32:0] m;         // 扩展的乘数，带额外位实现 Booth 编码
    reg [5:0]  count;     // 迭代次数
    reg        busy;      // 乘法进行中标志

    // 初始化和执行 Booth 乘法
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            acc   <= 64'd0;
            m     <= 33'd0;
            count <= 6'd0;
            busy  <= 1'b0;
            product <= 64'd0;
        end else if (start) begin
            acc   <= {32'd0, multiplicand}; // 初始累加器
            m     <= {multiplier, 1'b0};    // 扩展乘数
            count <= 6'd32;                // 32 次循环
            busy  <= 1'b1;                 // 标志乘法进行中
        end else if (busy) begin
            case (m[1:0])
                2'b01: acc <= acc + {multiplicand, 32'd0};  // 加法操作
                2'b10: acc <= acc - {multiplicand, 32'd0};  // 减法操作
                default: ;
            endcase

            m     <= {acc[0], m[32:1]}; // 右移扩展乘数
            acc   <= {acc[63], acc[63:1]}; // 右移累加器，符号位扩展

            count <= count - 1;         // 计数减 1
            if (count == 6'd0) begin
                busy    <= 1'b0;        // 乘法完成
                product <= acc;         // 输出最终结果
            end
        end
    end

    assign done = !busy; // busy 为 0 时，乘法完成

endmodule
