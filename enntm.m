
 function tm = enntm(n_hid,n_ens,lat,h,y,m,d,t_mgtrop,ts,pw,enntm_weights)

% enntm: Enhanced neural network model (ENNTm) for computing weighted mean temperature (Tm)
% in GNSS meteorological applications.
%
% 
% 
% (c) School of Transportation, Southeast University, 16 January 2021.
% Author: Fengyang Long
% 
% 
% 
% input parameters:
% n_hid   is the number of neurons in the hidden layer of a BPNN, values of
%             5,10,15,...,45,50. (scalar, indispensable)
% n_ens  is the ensemble size, values of 1,2,...,50. (scalar, indispensable)
% lat       is the latitude in degree. (scalar, indispensable)
% h         is the height (above mean sea level) of the site in metre
%            (scalar, indispensable), it can be converted from the ellipsoid height
%            via EGM2008 model provided by MATLAB, for example, h =
%            hgt-geoidheight(lat,lon,'egm2008','error'), hgt is the ellipsoid height.
% y         is the year. (scalar, indispensable)
% m       is the month. (scalar, indispensable)
% d        is the date. (scalar, indispensable)
% t_mgtrop is the Tm value computed by GTrop model, the matlab source code can be
%           found at https://github.com/sun1753814280/GTrop, refer to the paper:
%           Sun, Z.; Zhang, B.; Yao, Y. A Global Model for Estimating Tropospheric Delay and 
%           Weighted Mean Temperature Developed with Atmospheric Reanalysis Data from 
%           1979 to 2017. Remote Sens. 2019, 11, 1893. https://doi.org/10.3390/rs11161893
% ts       is the air temperature of the site in K. (scalar, indispensable for ENNTm-B and ENNTm-C)
% pw     is the water vapor pressure of the site in hPa. (scalar, , indispensable for ENNTm-C)
%           enntm_weights is the weight and bias values after training, A three
%           dimensional matrix(10¡Á50¡Á3), enntm_weights(m,n,k), where m 
%           BPNN structures, n traning missions, k (1. ENNTm-A; 2. ENNTm-B; 3.ENNTm-C).
% 
% 
% output parameter:
% tm     is the weighted mean temerature of the site in K. (scalar)


% -----------------------------------------------------------------
% compute the doy (day of year)
doy = date2doy(y,m,d);
% -----------------------------------------------------------------
% read the weights and bias values of ENNTm
load('enntm_weights.mat');

% -----------------------------------------------------------------
 % computing Tm with ENNTm 
if nargin == 9
    % ENNTm-A
    inputm = [h lat doy t_mgtrop]';
    tm_sum = 0;
    for i = 1:n_ens
        inps       = enntm_weights(n_hid/5,i,1).inputps; % The normalized parameters    
        oups      = enntm_weights(n_hid/5,i,1).outputps; %The inverse normalized parameters

        b1         = enntm_weights(n_hid/5,i,1).weights.input_b; % The bias values from input to hidden
        IW1_1   = enntm_weights(n_hid/5,i,1).weights.input_w; % The weights from input to hidden
        b2         = enntm_weights(n_hid/5,i,1).weights.hidden_b;% The bias values from hidden to output
        LW2_1  = enntm_weights(n_hid/5,i,1).weights.hidden_w;% The weights from hidden to output

        xn         = mapminmax('apply',inputm,inps);
        xh         = tansig_apply(repmat(b1,1,1)+IW1_1*xn);
        xp         = repmat(b2,1,1)+LW2_1*xh;
        tm1        = mapminmax('reverse',xp,oups);
        tm_sum = tm_sum + tm1;
    end
    tm = tm_sum/n_ens;
elseif nargin == 10
    % ENNTm-B
        inputm = [h lat doy t_mgtrop ts]';
        tm_sum = 0;
        for i = 1:n_ens
            inps       = enntm_weights(n_hid/5,i,2).inputps; % The normalized parameters    
            oups      = enntm_weights(n_hid/5,i,2).outputps; %The inverse normalized parameters

            b1         = enntm_weights(n_hid/5,i,2).weights.input_b; % The bias values from input to hidden
            IW1_1   = enntm_weights(n_hid/5,i,2).weights.input_w; % The weights from input to hidden
            b2         = enntm_weights(n_hid/5,i,2).weights.hidden_b;% The bias values from hidden to output
            LW2_1  = enntm_weights(n_hid/5,i,2).weights.hidden_w;% The weights from hidden to output

            xn         = mapminmax('apply',inputm,inps);
            xh         = tansig_apply(repmat(b1,1,1)+IW1_1*xn);
            xp         = repmat(b2,1,1)+LW2_1*xh;
            tm1        = mapminmax('reverse',xp,oups);
            tm_sum = tm_sum + tm1;
        end
        tm = tm_sum/n_ens;
elseif nargin == 11
    % ENNTm-C
        inputm = [h lat doy t_mgtrop ts pw]';
        tm_sum = 0;
        for i = 1:n_ens
            inps       = enntm_weights(n_hid/5,i,3).inputps;    
            oups      = enntm_weights(n_hid/5,i,3).outputps; 

            b1         = enntm_weights(n_hid/5,i,3).weights.input_b;
            IW1_1   = enntm_weights(n_hid/5,i,3).weights.input_w;
            b2         = enntm_weights(n_hid/5,i,3).weights.hidden_b;
            LW2_1  = enntm_weights(n_hid/5,i,3).weights.hidden_w;

            xn         = mapminmax('apply',inputm,inps);
            xh         = tansig_apply(repmat(b1,1,1)+IW1_1*xn);
            xp         = repmat(b2,1,1)+LW2_1*xh;
            tm1        = mapminmax('reverse',xp,oups);
            tm_sum = tm_sum + tm1;
        end
        tm = tm_sum/n_ens;
else
    error('Input error !');
end

 end
        
% -----------------------------------------------------------------
%The function to compute doy
function doy = date2doy(year,month,day)
doy = floor(month*275/9)-floor((month+9)/12).*(floor((year-4*floor(year/4)+2)/3)+1)+day-30;
end
% -----------------------------------------------------------------
%The activation function to connect the input layer and hidden layer
function a = tansig_apply(n,~)
a = 2 ./ (1 + exp(-2*n)) - 1;
end

