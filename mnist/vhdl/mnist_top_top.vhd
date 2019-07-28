----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 07/03/2019 02:37:52 PM
-- Design Name: 
-- Module Name: top_module - Behavioral
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

entity mnist_top_top is
    Port ( serial_in : in STD_LOGIC;
           fin_out : out STD_LOGIC_VECTOR (79 downto 0);
           clr : in std_logic;
			  rst :in std_logic;
           clk : in STD_LOGIC);
end mnist_top_top;

architecture Behavioral of mnist_top_top is
component mnist_org
    Port ( inp_feat : in  STD_LOGIC_VECTOR (511 downto 0);
           out_fin : out  STD_LOGIC_VECTOR (79 downto 0));
end component;

component shift_reg is
    Port ( serial_in : in STD_LOGIC;
           clk : in STD_LOGIC;
           clr : in std_logic;
           parallel_out : out STD_LOGIC_VECTOR (511 downto 0));
end component;
signal connect_wire : std_logic_vector(511 downto 0);
signal false_out: std_logic_vector(79 downto 0);

begin
full_imp_inst :mnist_org port map(inp_feat => connect_wire, out_fin => false_out);
shift_reg_inst : shift_reg port map(serial_in => serial_in, clk => clk, clr => clr, parallel_out => connect_wire); 

process(clk,rst,false_out)
begin
if rst = '1' then
		fin_out <= (others => '0');
else
		if rising_edge(clk) then
			fin_out <= false_out;
		end if;
end if;
end process;


end Behavioral;
