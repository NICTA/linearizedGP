% Function to generate 3/5 USPS data for the extended/unscented GPs
%
% Author: 	    Daniel Steinberg
% Institute: 	NICTA
% Date:		    9 Oct 2014

[x, y, xx, yy] = loadBinaryUSPS(3, 5);

save('USPS_3_5_data.mat', 'x', 'xx', 'y', 'yy', '-v6');
