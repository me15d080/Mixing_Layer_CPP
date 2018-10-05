clc; close all; clear all;
% get the data from out files
N = 50*100+1; % no. of time steps = 50
for i = 1:N
  string  = sprintf('/home/vikas/Desktop/mixinglayer_cpp/64out_o3upwind/%d/omg', i-1);
  OMG_CELL {i}=load(string,'-ascii');
end
xx = linspace(-1,1,64);% nx=128
x  = xx(1:end-1);
% calculate the vorticity thickness
sigma = zeros(1,N);
for j = 1:N
omg = OMG_CELL{j};
sig = zeros(1,64);
    for i = 1:64                      % ny
        from_here = 1+63*(i-1);       % nxc=nx-1
        to_here = 63*i;
        disp(omg(from_here:to_here));
        I=trapz(x',omg(from_here:to_here));
        sig(i)=I;
    end
sig0=max(abs(sig));
sigma(j) = 56/sig0;
end

% write vorticity thickness data
fid=fopen('sigma.txt','w');
for i=1:N
fprintf(fid, '%f \n', sigma(i));
end
fclose(fid);
% plot  vorticity thickness data
tu=linspace(0,50,N);
f1 = figure(1);
W = 5; H = 3;
set(f1,'PaperUnits','inches');set(f1,'PaperOrientation','portrait');
set(f1,'PaperSize',[H,W])    ;set(f1,'PaperPosition',[0,0,W,H]);
plot(tu,sigma);axis([0,50,0,10]);
xlabel('time units');ylabel('relative vorticity thickness');
print(f1,'-deps','vorticity_thickness.eps');
