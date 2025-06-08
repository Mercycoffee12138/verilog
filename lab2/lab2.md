# <center>计算机组成原理实验报告<center>

**实验名称：4个数加和器   班级：张金老师   姓名：王众  学号：2313211**

**实验老师：董前琨   实验地点：实验楼306 实验时间：2025.3.16**

## 一、实验目的

1. 理解定点乘法的不同实现算法的原理，掌握基本实现算法。
2.  熟悉并运用verilog语言进行电路设计。 
3. 为后续设计cpu的实验打下基础。

## 二、实验内容说明

### 1.复现32位乘法器

​	复现指导书中的32位乘法器，使用两个32位寄存器，将其中的内容相乘，存储在一个64位寄存器 中，并在实验箱上测试。

### 2.改进乘法器

​	要求在初始的乘法器的基础上进行改进，让算法能够在16个时钟节拍之内完成，提高乘法效率。上 箱验证更改后的代码是否仍然正确。

## 三、实验原理图

![image-20250319191236383](C:\Users\coffe\AppData\Roaming\Typora\typora-user-images\image-20250319191236383.png)

​	整体的实验原理图与模板相同，只是在迭代乘法处修改了一部分内容，下面是我们对于迭代乘法中 的具体修改。![f1c806aa5b3dc9c67382868b44e6618](D:\WeChat Files\wxid_dgezzh4ieb0q12\FileStorage\Temp\f1c806aa5b3dc9c67382868b44e6618.png)

## 四、实验步骤

### 1.multiply.v模块的修改

#### 模块功能

​	本模块是用来实现两个数的乘法运算，我们需要对源代码进行修改，使得每次移动的乘数和输出的结果都从一位变为两位，从而实现时钟周期的缩短。

#### 代码部分的修改

​	为了实现每次乘法运算时移动的位数变为两位，所以我们要在`multplicand`模块进行更改。

```verilog
multiplicand <= {multiplicand[61:0],2'b00};
```

​	原来的代码实现是将被乘数的前62位取出，再将最低位的一位设置为0，这样就完成了移动一位的操作。现在我们将其修改为去除被乘数的前61位，再将最后两位设置为0，实现左移两位的操作。

```verilog
multiplier <= {2'b00,multiplier[31:2]}; 
```

​	因为每次乘数都要移动两位，所以我们的分步乘法操作也要进行相应的修改，即从之前的考虑0和1到考虑1到3（00，01，10，11）四种情况。所以我们将原来的二路选择器变为现在的四路选择器：

```verilog
wire [63:0] partial_product1;
wire [63:0] partial_product2;
assign partial_product1 = multiplier[0] ? multiplicand : 64'd0;
assign partial_product2 = multiplier[1] ? multiplicand : 64'd0;
```

​	我们使用了两个临时变量来存储我们的临时乘积，`partial_product1`代表着我们乘数的低位，如果乘数为0则赋值为0不为0则赋值为`multiplicand`。`partial_product2`代表着乘数的高位，取值和`product1`相同但是需要在加和时进行乘2操作。这样保证了位置和值的统一。

```verilog
product_temp <= product_temp + partial_product1+2*partial_product2;
```

最后我们对最终的赋值语句进行修改，将临时变量`tmp`里面的值与两个`product`相加即可。

​	我们来做进一步的解释，把两个 partial_product均累加到 partial_temp上。其中当最低两位为00 ，partial_temp加上0，即0倍的被乘数；当最低两位为01时，partial_product1保存着1倍的被乘数而 partial_product2为0，实现了对partial_temp加上1倍的被乘数；当最低两位为10时，partial_product1 为0而partial_product2保存着2倍的被乘数，实现对partial_temp加上 2倍的被乘数；当最低两位为11 时，partial_product1保存1倍的被乘数， partial_product2保存2倍的被乘数，实现对partial_temp加上 3倍的被乘数。通过以上的分析，我们就可以发现，在上述修改中我们完成了对乘法器的修改。

#### 2._display模块和约束文件

因为修改的是乘法器内部的逻辑，所以对于其他的模块没有需要改进的地方。

## 五、实验结果分析

### 仿真模拟

经验证乘法结果是正确的.

1111*1111=1234321

![](D:\WeChat Files\wxid_dgezzh4ieb0q12\FileStorage\Temp\b40654dc5dcd84c5f8c6f409915d431.png)

### 上箱验证：

（1）正数相乘的普通乘法

![image-20250319194941714](C:\Users\coffe\AppData\Roaming\Typora\typora-user-images\image-20250319194941714.png)

（2）-1*5=-5

![image-20250319195019125](C:\Users\coffe\AppData\Roaming\Typora\typora-user-images\image-20250319195019125.png)

（3）-4*-6=24

![image-20250319195056322](C:\Users\coffe\AppData\Roaming\Typora\typora-user-images\image-20250319195056322.png)

## 六、改进方案2

### 使用Booth算法改进乘法器

**Booth 算法**是一种高效的**二进制乘法**方法，能够处理**有符号数**并且**加速乘法过程**。它通过分析乘数的编码模式，减少不必要的加法和减法操作，从而提升计算效率。

`multiply.v`文件

```verilog
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

```

`multiply_display.v`文件

```verilog
module adder_display (
    input  wire        clk,
    input  wire        resetn,
    input  wire        start,
    input  wire [31:0] multiplicand,
    input  wire [31:0] multiplier,

    // LCD 显示接口
    output reg         display_valid,
    output reg  [39:0] display_name,
    output reg  [31:0] display_value,
    input  wire [5:0]  display_number,

    output wire        done
);

    // 乘法结果
    wire [63:0] product;

    // 调用 Booth 乘法器模块
    adder adder_inst (
        .clk(clk),
        .resetn(resetn),
        .start(start),
        .multiplicand(multiplicand),
        .multiplier(multiplier),
        .product(product),
        .done(done)
    );

    // LCD 显示逻辑
    always @(posedge clk or negedge resetn) begin
        if (!resetn) begin
            display_valid <= 1'b0;
            display_name  <= 40'd0;
            display_value <= 32'd0;
        end else begin
            case (display_number)
                6'd1: begin
                    display_valid <= 1'b1;
                    display_name  <= "MUL1";
                    display_value <= multiplicand;
                end
                6'd2: begin
                    display_valid <= 1'b1;
                    display_name  <= "MUL2";
                    display_value <= multiplier;
                end
                6'd3: begin
                    display_valid <= 1'b1;
                    display_name  <= "PRO_H";
                    display_value <= product[63:32];
                end
                6'd4: begin
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
    end

endmodule

```

经验证符合我们的实验预期。

## 七、总结感想

1.在本次实验中,进一步了解了vivado和verilog语句的认识,对数据的唯一进行了实践操作并有了比较好的应用与理解,也对乘法器的原理和硬件实现有了进一步的了解.

2.对于时钟周期的优化，我们要考虑到移动位数的变化以及通过位数的改变能否影响到具体的时钟周 期，我们通过将32个时钟周期长度进行一次性移动两位，实现了时钟周期的缩短，完成了优化。

3.我会进一步查询资料去了解有没有更加快速的优化方式.去进一步优化乘法器的效率.
