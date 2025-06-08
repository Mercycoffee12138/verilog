module adder_display (
    input  wire        clk,
    input  wire        resetn,
    input  wire        input_sel,     // 选择输入：0-乘数1，1-乘数2
    input  wire        sw_begin,      // 乘法开始信号
    input  wire [31:0] input_value,   // 触摸屏输入值

    output wire        lcd_rst,
    output wire        lcd_cs,
    output wire        lcd_rs,
    output wire        lcd_wr,
    output wire        lcd_rd,
    inout  wire [15:0] lcd_data_io,
    output wire        lcd_bl_ctr,
    inout  wire        ct_int,
    inout  wire        ct_sda,
    output wire        ct_scl,
    output wire        ct_rstn
);

    reg [31:0] mult_op1, mult_op2;
    wire [63:0] product;
    wire        mult_done;

    always @(posedge clk) begin
        if (!resetn) begin
            mult_op1 <= 32'd0;
            mult_op2 <= 32'd0;
        end else if (input_sel == 1'b0) begin
            mult_op1 <= input_value; // 选择第一个乘数
        end else if (input_sel == 1'b1) begin
            mult_op2 <= input_value; // 选择第二个乘数
        end
    end

    // Booth 乘法器实现
    reg [63:0] acc;
    reg [32:0] m;
    reg [5:0]  count;
    reg        busy;

    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            acc   <= 64'd0;
            m     <= 33'd0;
            count <= 6'd0;
            busy  <= 1'b0;
        end else if (sw_begin) begin
            acc   <= {32'd0, mult_op1};
            m     <= {mult_op2, 1'b0};
            count <= 6'd32;
            busy  <= 1'b1;
        end else if (busy) begin
            case (m[1:0])
                2'b01: acc <= acc + {mult_op1, 32'd0};
                2'b10: acc <= acc - {mult_op1, 32'd0};
                default: ;
            endcase
            m <= {acc[0], m[32:1]};
            acc <= {acc[63], acc[63:1]};
            count <= count - 1;
            if (count == 0) busy <= 1'b0;
        end
    end

    assign product = acc;
    assign mult_done = !busy;

    reg display_valid;
    reg [39:0] display_name;
    reg [31:0] display_value;
    wire [5:0] display_number;

    lcd_module lcd_module (
        .clk(clk),
        .resetn(resetn),
        .display_valid(display_valid),
        .display_name(display_name),
        .display_value(display_value),
        .display_number(display_number),
        .lcd_rst(lcd_rst),
        .lcd_cs(lcd_cs),
        .lcd_rs(lcd_rs),
        .lcd_wr(lcd_wr),
        .lcd_rd(lcd_rd),
        .lcd_data_io(lcd_data_io),
        .lcd_bl_ctr(lcd_bl_ctr),
        .ct_int(ct_int),
        .ct_sda(ct_sda),
        .ct_scl(ct_scl),
        .ct_rstn(ct_rstn)
    );

    always @(posedge clk) begin
        case (display_number)
            6'd5: begin
                display_valid <= 1'b1;
                display_name  <= "M_OP1";
                display_value <= mult_op1;
            end
            6'd6: begin
                display_valid <= 1'b1;
                display_name  <= "M_OP2";
                display_value <= mult_op2;
            end
            6'd7: begin
                display_valid <= 1'b1;
                display_name  <= "PRO_H";
                display_value <= product[63:32];
            end
            6'd8: begin
                display_valid <= 1'b1;
                display_name  <= "PRO_L";
                display_value <= product[31:0];
            end
            default: begin
                display_valid <= 1'b0;
                display_name  <= 40'd0;
                display_value <= 32'd0;
            end
        endcase
    end

endmodule
