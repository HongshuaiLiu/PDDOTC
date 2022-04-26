clear
close all

miu1_inf=0.01;
deta1=1;
deta2=1;

Tf=20;
T=0.01;
t=0:T:Tf;
N=Tf/T;
V=zeros(1,N+1);
X=zeros(4,N+1);

Wc=zeros(18,N+1);
fai=zeros(18,N+1);
Wu=zeros(14,N+1);
theta=zeros(14,N+1);
u=zeros(1,N+1);
R=1;
Qe=100*[1 0;0 1];
Q=[Qe(1,:) 0 0;Qe(2,:) 0 0;0 0 0 0;0 0 0 0];
lambta=0.01;


Wu(:,1)=[-9.88398492963319;-12.2266175854492;-0.476529297233046;0.559512796878357;0.00528822266644529;0.0668961902489680;0.0287942642610225;-0.242486760350069;0.176453536692838;0.0983851802804582;-0.0345184274175016;-0.227805469468572;0.141757508037304;0.155334613438680];
Wc(:,1)=[66.3860250378709;30.8054836712642;-3.76343108900821;-0.115682548226501;17.3051655969843;1.41124547449646;-2.12038861368573;13.5490693714610;-1.90087662815591;11.4624758033469;-0.0784973248707611;0.446715788674581;0.225825849960141;0.809004451740171;-16.5708325646672;-9.64735679923409;-5.05231198935032;1.84618190198355];
X(:,1)=[0.1 -0.4 0 0.5]';
%%

ST=1;
k=1;
l=1;
for i=ST:N
 miu1(i)=csch(k,i*T,l)+miu1_inf;%PPF
if  X(1,i)>=miu1(i)
    temp1=4;
elseif  X(1,i)<=(-miu1(i))
    temp1=-4;
else
    temp1=0.5*log((X(1,i)+deta1*miu1(i))./(deta2*miu1(i)-X(1,i)));
end

Z(:,i)=[temp1;X(2,i);X(3,i);X(4,i)];
fai(:,i)=Fai(Z(:,i)); 
V(i)=fai(:,i)'*Wc(:,1);  
theta(:,i)=Theta(Z(:,i));
u(i)=theta(:,i)'*Wu(:,1);   
X(:,i+1)=X(:,i)+T*zengguang(X(:,i),u(i)); 
end
figure
plot(X(3,1:2000))
hold on
plot(X(1,1:2000)+X(3,1:2000))
legend('Reference','system output')

figure
plot(X(1,1:2000),'r')
hold on
plot(miu1,'k')
plot(-miu1,'k')
legend('tracking error','preassigned bound')


%% AUGMENTED SYSTEM
function Xdot = zengguang(X,u)
A=[0 1 0 0; -0.5 -0.5 0.5 -0.5; 0 0 0 1; 0 0 -1 0];
B=[0 1 0 0]';
temp=[0 0.5*(X(1)+X(3))^2*(X(2)+X(4)) 0 0]';
Xdot=A*X+temp+B*u;
end

%activation function
function f=Fai(X)
 f=[X(1)^2,X(1)*X(2),X(1)*X(3),X(1)*X(4), X(2)^2,X(2)*X(3),X(2)*X(4),X(3)^2,X(3)*X(4),X(4)^2,X(1)^3,X(2)^3,X(3)^3,X(4)^3,X(1)^4,X(2)^4,X(3)^4,X(4)^4]';
end
%activation function
function f=Theta(X)
 f=[X(1),X(2),X(3),X(4),X(1)^2,X(1)*X(2),X(1)*X(3),X(1)*X(4),X(2)^2,X(2)*X(3),X(2)*X(4),X(3)^2,X(3)*X(4),X(4)^2]';
end
%transformation function
function f=csch(k,t,l)
temp=k*t+l;
f=2/(exp(temp)-exp(-temp));
end

