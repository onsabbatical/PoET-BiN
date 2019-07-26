----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 07/08/2019 11:29:21 PM
-- Design Name: 
-- Module Name: LUT8 - Behavioral
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

entity LUT8 is
generic(INIT : std_logic_vector(255 downto 0) := (others => '0') );
port(I0:in std_logic;
I1:in std_logic;
I2:in std_logic;
I3:in std_logic;
I4:in std_logic;
I5:in std_logic;
I6:in std_logic;
I7:in std_logic;
O:out std_logic);
--  Port ( );
end LUT8;

architecture Behavioral of LUT8 is
signal all_inp : std_logic_vector(7 downto 0);
begin
all_inp <= I7 & I6 & I5 & I4 & I3 & I2 & I1 & I0;
pmux : process(all_inp)
begin
case all_inp is
	 when "00000000" => O <= INIT(0); 
	 when "00000001" => O <= INIT(1); 
	 when "00000010" => O <= INIT(2); 
	 when "00000011" => O <= INIT(3); 
	 when "00000100" => O <= INIT(4); 
	 when "00000101" => O <= INIT(5); 
	 when "00000110" => O <= INIT(6); 
	 when "00000111" => O <= INIT(7); 
	 when "00001000" => O <= INIT(8); 
	 when "00001001" => O <= INIT(9); 
	 when "00001010" => O <= INIT(10); 
	 when "00001011" => O <= INIT(11); 
	 when "00001100" => O <= INIT(12); 
	 when "00001101" => O <= INIT(13); 
	 when "00001110" => O <= INIT(14); 
	 when "00001111" => O <= INIT(15); 
	 when "00010000" => O <= INIT(16); 
	 when "00010001" => O <= INIT(17); 
	 when "00010010" => O <= INIT(18); 
	 when "00010011" => O <= INIT(19); 
	 when "00010100" => O <= INIT(20); 
	 when "00010101" => O <= INIT(21); 
	 when "00010110" => O <= INIT(22); 
	 when "00010111" => O <= INIT(23); 
	 when "00011000" => O <= INIT(24); 
	 when "00011001" => O <= INIT(25); 
	 when "00011010" => O <= INIT(26); 
	 when "00011011" => O <= INIT(27); 
	 when "00011100" => O <= INIT(28); 
	 when "00011101" => O <= INIT(29); 
	 when "00011110" => O <= INIT(30); 
	 when "00011111" => O <= INIT(31); 
	 when "00100000" => O <= INIT(32); 
	 when "00100001" => O <= INIT(33); 
	 when "00100010" => O <= INIT(34); 
	 when "00100011" => O <= INIT(35); 
	 when "00100100" => O <= INIT(36); 
	 when "00100101" => O <= INIT(37); 
	 when "00100110" => O <= INIT(38); 
	 when "00100111" => O <= INIT(39); 
	 when "00101000" => O <= INIT(40); 
	 when "00101001" => O <= INIT(41); 
	 when "00101010" => O <= INIT(42); 
	 when "00101011" => O <= INIT(43); 
	 when "00101100" => O <= INIT(44); 
	 when "00101101" => O <= INIT(45); 
	 when "00101110" => O <= INIT(46); 
	 when "00101111" => O <= INIT(47); 
	 when "00110000" => O <= INIT(48); 
	 when "00110001" => O <= INIT(49); 
	 when "00110010" => O <= INIT(50); 
	 when "00110011" => O <= INIT(51); 
	 when "00110100" => O <= INIT(52); 
	 when "00110101" => O <= INIT(53); 
	 when "00110110" => O <= INIT(54); 
	 when "00110111" => O <= INIT(55); 
	 when "00111000" => O <= INIT(56); 
	 when "00111001" => O <= INIT(57); 
	 when "00111010" => O <= INIT(58); 
	 when "00111011" => O <= INIT(59); 
	 when "00111100" => O <= INIT(60); 
	 when "00111101" => O <= INIT(61); 
	 when "00111110" => O <= INIT(62); 
	 when "00111111" => O <= INIT(63); 
	 when "01000000" => O <= INIT(64); 
	 when "01000001" => O <= INIT(65); 
	 when "01000010" => O <= INIT(66); 
	 when "01000011" => O <= INIT(67); 
	 when "01000100" => O <= INIT(68); 
	 when "01000101" => O <= INIT(69); 
	 when "01000110" => O <= INIT(70); 
	 when "01000111" => O <= INIT(71); 
	 when "01001000" => O <= INIT(72); 
	 when "01001001" => O <= INIT(73); 
	 when "01001010" => O <= INIT(74); 
	 when "01001011" => O <= INIT(75); 
	 when "01001100" => O <= INIT(76); 
	 when "01001101" => O <= INIT(77); 
	 when "01001110" => O <= INIT(78); 
	 when "01001111" => O <= INIT(79); 
	 when "01010000" => O <= INIT(80); 
	 when "01010001" => O <= INIT(81); 
	 when "01010010" => O <= INIT(82); 
	 when "01010011" => O <= INIT(83); 
	 when "01010100" => O <= INIT(84); 
	 when "01010101" => O <= INIT(85); 
	 when "01010110" => O <= INIT(86); 
	 when "01010111" => O <= INIT(87); 
	 when "01011000" => O <= INIT(88); 
	 when "01011001" => O <= INIT(89); 
	 when "01011010" => O <= INIT(90); 
	 when "01011011" => O <= INIT(91); 
	 when "01011100" => O <= INIT(92); 
	 when "01011101" => O <= INIT(93); 
	 when "01011110" => O <= INIT(94); 
	 when "01011111" => O <= INIT(95); 
	 when "01100000" => O <= INIT(96); 
	 when "01100001" => O <= INIT(97); 
	 when "01100010" => O <= INIT(98); 
	 when "01100011" => O <= INIT(99); 
	 when "01100100" => O <= INIT(100); 
	 when "01100101" => O <= INIT(101); 
	 when "01100110" => O <= INIT(102); 
	 when "01100111" => O <= INIT(103); 
	 when "01101000" => O <= INIT(104); 
	 when "01101001" => O <= INIT(105); 
	 when "01101010" => O <= INIT(106); 
	 when "01101011" => O <= INIT(107); 
	 when "01101100" => O <= INIT(108); 
	 when "01101101" => O <= INIT(109); 
	 when "01101110" => O <= INIT(110); 
	 when "01101111" => O <= INIT(111); 
	 when "01110000" => O <= INIT(112); 
	 when "01110001" => O <= INIT(113); 
	 when "01110010" => O <= INIT(114); 
	 when "01110011" => O <= INIT(115); 
	 when "01110100" => O <= INIT(116); 
	 when "01110101" => O <= INIT(117); 
	 when "01110110" => O <= INIT(118); 
	 when "01110111" => O <= INIT(119); 
	 when "01111000" => O <= INIT(120); 
	 when "01111001" => O <= INIT(121); 
	 when "01111010" => O <= INIT(122); 
	 when "01111011" => O <= INIT(123); 
	 when "01111100" => O <= INIT(124); 
	 when "01111101" => O <= INIT(125); 
	 when "01111110" => O <= INIT(126); 
	 when "01111111" => O <= INIT(127); 
	 when "10000000" => O <= INIT(128); 
	 when "10000001" => O <= INIT(129); 
	 when "10000010" => O <= INIT(130); 
	 when "10000011" => O <= INIT(131); 
	 when "10000100" => O <= INIT(132); 
	 when "10000101" => O <= INIT(133); 
	 when "10000110" => O <= INIT(134); 
	 when "10000111" => O <= INIT(135); 
	 when "10001000" => O <= INIT(136); 
	 when "10001001" => O <= INIT(137); 
	 when "10001010" => O <= INIT(138); 
	 when "10001011" => O <= INIT(139); 
	 when "10001100" => O <= INIT(140); 
	 when "10001101" => O <= INIT(141); 
	 when "10001110" => O <= INIT(142); 
	 when "10001111" => O <= INIT(143); 
	 when "10010000" => O <= INIT(144); 
	 when "10010001" => O <= INIT(145); 
	 when "10010010" => O <= INIT(146); 
	 when "10010011" => O <= INIT(147); 
	 when "10010100" => O <= INIT(148); 
	 when "10010101" => O <= INIT(149); 
	 when "10010110" => O <= INIT(150); 
	 when "10010111" => O <= INIT(151); 
	 when "10011000" => O <= INIT(152); 
	 when "10011001" => O <= INIT(153); 
	 when "10011010" => O <= INIT(154); 
	 when "10011011" => O <= INIT(155); 
	 when "10011100" => O <= INIT(156); 
	 when "10011101" => O <= INIT(157); 
	 when "10011110" => O <= INIT(158); 
	 when "10011111" => O <= INIT(159); 
	 when "10100000" => O <= INIT(160); 
	 when "10100001" => O <= INIT(161); 
	 when "10100010" => O <= INIT(162); 
	 when "10100011" => O <= INIT(163); 
	 when "10100100" => O <= INIT(164); 
	 when "10100101" => O <= INIT(165); 
	 when "10100110" => O <= INIT(166); 
	 when "10100111" => O <= INIT(167); 
	 when "10101000" => O <= INIT(168); 
	 when "10101001" => O <= INIT(169); 
	 when "10101010" => O <= INIT(170); 
	 when "10101011" => O <= INIT(171); 
	 when "10101100" => O <= INIT(172); 
	 when "10101101" => O <= INIT(173); 
	 when "10101110" => O <= INIT(174); 
	 when "10101111" => O <= INIT(175); 
	 when "10110000" => O <= INIT(176); 
	 when "10110001" => O <= INIT(177); 
	 when "10110010" => O <= INIT(178); 
	 when "10110011" => O <= INIT(179); 
	 when "10110100" => O <= INIT(180); 
	 when "10110101" => O <= INIT(181); 
	 when "10110110" => O <= INIT(182); 
	 when "10110111" => O <= INIT(183); 
	 when "10111000" => O <= INIT(184); 
	 when "10111001" => O <= INIT(185); 
	 when "10111010" => O <= INIT(186); 
	 when "10111011" => O <= INIT(187); 
	 when "10111100" => O <= INIT(188); 
	 when "10111101" => O <= INIT(189); 
	 when "10111110" => O <= INIT(190); 
	 when "10111111" => O <= INIT(191); 
	 when "11000000" => O <= INIT(192); 
	 when "11000001" => O <= INIT(193); 
	 when "11000010" => O <= INIT(194); 
	 when "11000011" => O <= INIT(195); 
	 when "11000100" => O <= INIT(196); 
	 when "11000101" => O <= INIT(197); 
	 when "11000110" => O <= INIT(198); 
	 when "11000111" => O <= INIT(199); 
	 when "11001000" => O <= INIT(200); 
	 when "11001001" => O <= INIT(201); 
	 when "11001010" => O <= INIT(202); 
	 when "11001011" => O <= INIT(203); 
	 when "11001100" => O <= INIT(204); 
	 when "11001101" => O <= INIT(205); 
	 when "11001110" => O <= INIT(206); 
	 when "11001111" => O <= INIT(207); 
	 when "11010000" => O <= INIT(208); 
	 when "11010001" => O <= INIT(209); 
	 when "11010010" => O <= INIT(210); 
	 when "11010011" => O <= INIT(211); 
	 when "11010100" => O <= INIT(212); 
	 when "11010101" => O <= INIT(213); 
	 when "11010110" => O <= INIT(214); 
	 when "11010111" => O <= INIT(215); 
	 when "11011000" => O <= INIT(216); 
	 when "11011001" => O <= INIT(217); 
	 when "11011010" => O <= INIT(218); 
	 when "11011011" => O <= INIT(219); 
	 when "11011100" => O <= INIT(220); 
	 when "11011101" => O <= INIT(221); 
	 when "11011110" => O <= INIT(222); 
	 when "11011111" => O <= INIT(223); 
	 when "11100000" => O <= INIT(224); 
	 when "11100001" => O <= INIT(225); 
	 when "11100010" => O <= INIT(226); 
	 when "11100011" => O <= INIT(227); 
	 when "11100100" => O <= INIT(228); 
	 when "11100101" => O <= INIT(229); 
	 when "11100110" => O <= INIT(230); 
	 when "11100111" => O <= INIT(231); 
	 when "11101000" => O <= INIT(232); 
	 when "11101001" => O <= INIT(233); 
	 when "11101010" => O <= INIT(234); 
	 when "11101011" => O <= INIT(235); 
	 when "11101100" => O <= INIT(236); 
	 when "11101101" => O <= INIT(237); 
	 when "11101110" => O <= INIT(238); 
	 when "11101111" => O <= INIT(239); 
	 when "11110000" => O <= INIT(240); 
	 when "11110001" => O <= INIT(241); 
	 when "11110010" => O <= INIT(242); 
	 when "11110011" => O <= INIT(243); 
	 when "11110100" => O <= INIT(244); 
	 when "11110101" => O <= INIT(245); 
	 when "11110110" => O <= INIT(246); 
	 when "11110111" => O <= INIT(247); 
	 when "11111000" => O <= INIT(248); 
	 when "11111001" => O <= INIT(249); 
	 when "11111010" => O <= INIT(250); 
	 when "11111011" => O <= INIT(251); 
	 when "11111100" => O <= INIT(252); 
	 when "11111101" => O <= INIT(253); 
	 when "11111110" => O <= INIT(254); 
	 when "11111111" => O <= INIT(255);
	 when others => O <= '0';
  end case;
end process pmux;

end Behavioral;
