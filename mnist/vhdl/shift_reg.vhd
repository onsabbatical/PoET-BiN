----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 07/03/2019 12:49:41 PM
-- Design Name: 
-- Module Name: shift_reg - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity shift_reg is
    Port ( serial_in : in STD_LOGIC;
           clk : in STD_LOGIC;
           clr : in std_logic;
           parallel_out : out STD_LOGIC_VECTOR (511 downto 0));
end shift_reg;

architecture Behavioral of shift_reg is
signal cout_shift : std_logic_vector(511 downto 0) := (others => '0');
component d_ff
Port ( d : in STD_LOGIC;
           q : out STD_LOGIC;
           clr : in std_logic;
           clk : in STD_LOGIC);
end component;
begin
d_ff_gen_1 : d_ff port map (d => serial_in, q => cout_shift(0), clk => clk, clr => clr);
gen_d_ff : for i in 1 to 511 generate
     d_ff_gen : d_ff port map (d => cout_shift(i-1), q => cout_shift(i), clk => clk, clr => clr);
end generate gen_d_ff;
parallel_out <= cout_shift;

end Behavioral;
