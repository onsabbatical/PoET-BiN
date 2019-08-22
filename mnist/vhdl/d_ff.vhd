----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 07/03/2019 01:09:46 PM
-- Design Name: 
-- Module Name: d_ff - Behavioral
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

entity d_ff is
    Port ( d : in STD_LOGIC;
           q : out STD_LOGIC;
           clr : in std_logic;
           clk : in STD_LOGIC);
end d_ff;

architecture Behavioral of d_ff is

begin
process(clk,clr)
begin 
     if(clr='1') then 
        q <= '0';
     elsif(rising_edge(clk)) then
        q <= d; 
     end if;      
 end process;


end Behavioral;
