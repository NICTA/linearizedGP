% linearizedGP -- Implementation of extended and unscented Gaussian processes.
% Copyright (C) 2014 National ICT Australia (NICTA)
%
% This file is part of linearizedGP.
%
% linearizedGP is free software: you can redistribute it and/or modify it under
% the terms of the GNU Lesser General Public License as published by the Free
% Software Foundation, either version 3 of the License, or (at your option) any
% later version.
%
% linearizedGP is distributed in the hope that it will be useful, but WITHOUT
% ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
% FOR A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
% details.
%
% You should have received a copy of the GNU Lesser General Public License
% along with linearizedGP. If not, see <http://www.gnu.org/licenses/>.

% Run the USPS handwritten digits experiment for the matlab algorithms (GPML
% toolbox) from our NIPS 2014 paper. This requires the GPML toolbox, and was
% originally run in octave.
%
% Author:     Daniel Steinberg (daniel.steinberg@nicta.com.au)
% Institute:  NICTA
% Date:       4 Sep 2014

% Settings
kinit = [1.0, 1.0];   % Initial kernel parameter settings
gpmlpath = '~/Code/matlab/gpml-matlab/startup.m'  % Path to GPML startup
datapath = '../data/USPS_3_5_data.mat'  % Path to data

% Load the GPML toolbox
run(gpmlpath)

% Load the USPS data
load(datapath)
pys = USPS.ys;
pys(pys == -1) = 0;

% GP setup
meanfunc = @meanConst;
covfunc = @covSEiso;

% Loss functions --------------------------------------------------------------

function nll = bernloglike(ys, pys)
    nll = sum(-(ys .* log(pys) + (1 - ys) .* log(1 - pys))) / length(ys);
end

function err = errrate(ys, pys)
    err = sum(ys ~= (pys >= 0.5)) / length(ys);
end


% Learn GPs -------------------------------------------------------------------

% Learn Laplace GP classifier
hyp_l.mean = 0;
hyp_l.cov = log(kinit);
hyp_l = minimize(hyp_l, @gp, -50, @infLaplace, meanfunc, covfunc, @likLogistic,
                 USPS.xr, USPS.yr);

% Learn EP GP classifier
hyp_e.mean = 0;
hyp_e.cov = log(kinit);
hyp_e = minimize(hyp_e, @gp, -50, @infEP, meanfunc, covfunc, @likLogistic,
                 USPS.xr, USPS.yr);

% Learn Variational GP classifier
hyp_v.mean = 0;
hyp_v.cov = log(kinit);
hyp_v = minimize(hyp_v, @gp, -50, @infVB, meanfunc, covfunc, @likLogistic,
                 USPS.xr, USPS.yr);


% Predictions -----------------------------------------------------------------

% Laplace
[ys_l, sys_l, ms_l, sms_l, pys_l] = gp(hyp_l, @infLaplace, meanfunc, covfunc,
                                       @likLogistic, USPS.xr, USPS.yr, USPS.xs,
                                       ones(size(USPS.ys)));
pys_l = exp(pys_l);

fprintf('Laplace: av nll = %f, err rate = %f\n', bernloglike(pys, pys_l),
        errrate(pys, pys_l));

% EP
[ys_e, sys_e, ms_e, sms_e, pys_e] = gp(hyp_e, @infEP, meanfunc, covfunc,
                                       @likLogistic, USPS.xr, USPS.yr, USPS.xs,
                                       ones(size(USPS.ys)));
pys_e = exp(pys_e);

fprintf('EP: av nll = %f, err rate = %f\n', bernloglike(pys, pys_e),
        errrate(pys, pys_e));

% VB 
[ys_v, sys_v, ms_v, sms_v, pys_v] = gp(hyp_v, @infVB, meanfunc, covfunc,
                                       @likLogistic, USPS.xr, USPS.yr, USPS.xs,
                                       ones(size(USPS.ys)));

pys_v = exp(pys_v);

fprintf('VB: av nll = %f, err rate = %f\n', bernloglike(pys, pys_v),
        errrate(pys, pys_v));
