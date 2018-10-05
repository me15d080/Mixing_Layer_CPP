%---Grid Independence Study Code---%
clc; close all; clear all;
%--our data--%
for i = 1:4
  string  = sprintf('/home/vikas/Desktop/mixinglayer_cpp/plot_files/o3_upwind/%d', i);
  SIG_CELL {i}=load(string,'-ascii');
end
tu_13 = linspace(0,50,5001);
tu_4  = linspace(0,50,10001);

f1 = figure(1);

sig = SIG_CELL{1};
plot(tu_13,sig,'-b','Linewidth',3);axis([0,50,0,4]);hold on;
sig = SIG_CELL{2};
plot(tu_13,sig,'-g','Linewidth',3);axis([0,50,0,4]);hold on;
sig = SIG_CELL{3};
plot(tu_13,sig,'-m','Linewidth',3);axis([0,50,0,4]);hold on;
sig = SIG_CELL{4};
plot( tu_4,sig,'-r','Linewidth',3);axis([0,50,0,4]);hold on;

%---DNS----%
filename = 'data.txt';
A=importdata(filename);
x=A(:,1);
y=A(:,2);

W = 5; H = 3;
set(f1,'PaperUnits','inches');set(f1,'PaperOrientation','portrait');
set(f1,'PaperSize',[H,W])    ;set(f1,'PaperPosition',[0,0,W,H]);
plot(x,y,'-k','Linewidth',3);axis([0,50,0,4]);hold off
xlabel('Time Units');ylabel('Relative Vorticity Thickness');
legend('u064','u128','u256','u512','DNS')
title('Grid Independence Study for COMPACT Upwind Scheme ')
print (f1,'-deps', '-color', 'COMPACT_Upwind.eps')




