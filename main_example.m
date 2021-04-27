% The example of computing Tm with ENNTm.
clc;clear

 
n_hid = 25;
n_ens = 50;
lat = -36.57;
lon = -64.27;
h = 7269.2;
y = 2016;
m = 1;
d = 1;
t_mgtrop = 244.32;
ts = 252.30;
pw = 0.319;

% ENNTm-A
tm1 = enntm(n_hid,n_ens,lat,h,y,m,d,t_mgtrop);
disp(['Tm value computed by ENNTm-A is: ',num2str(tm1,'%6.2f'),' K']);


% ENNTm-B
tm2 = enntm(n_hid,n_ens,lat,h,y,m,d,t_mgtrop,ts);
disp(['Tm value computed by ENNTm-B is: ',num2str(tm2,'%6.2f'),' K']);

% ENNTm-C
tm3 = enntm(n_hid,n_ens,lat,h,y,m,d,t_mgtrop,ts,pw);
disp(['Tm value computed by ENNTm-C is: ',num2str(tm3,'%6.2f'),' K']);




