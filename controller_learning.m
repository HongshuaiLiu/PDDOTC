clear
close all
miu1_ini=0.6;
deta1=1;
deta2=1;
yangben=1*(rand(1,800)-0.5);
yangben2=[yangben(1:200);yangben(201:400);yangben(401:600);yangben(601:800)];

  for k=1:200
 X(:,k)=[yangben2(1,k) yangben2(2,k) yangben2(3,k) yangben2(4,k)]'; 
if  X(1,k)>=miu1_ini
    temp1=4;
elseif  X(1,k)<=(-miu1_ini)
    temp1=-4;
else
    temp1=0.5*log((X(1,k)+deta1*miu1_ini)./(deta2*miu1_ini-X(1,k)));
end

Z(:,k)=[temp1;X(2,k);X(3,k);X(4,k)];
  end  

Tf=20;
T=0.1;
t=0:T:Tf;
N=Tf/T;
V=zeros(1,N+1);
X=zeros(4,N+1);
Xnext=zeros(4,N+1);
Wc=zeros(18,N+1);
fai=zeros(18,N+1);
Wu=zeros(14,N+1);
theta=zeros(14,N+1);
u=zeros(1,N+1);
Wcu=zeros(N+1,32);
R=1;
Qe=100*[1 0;0 1];
Q=[Qe(1,:) 0 0;Qe(2,:) 0 0;0 0 0 0;0 0 0 0];
lambta=0.01;
ST=200;
u(1:ST)=1*(rand(1,ST)-0.5);
for k=1:ST
X(:,k)=[yangben2(1,k) yangben2(2,k) yangben2(3,k) yangben2(4,k)]';
Xnext(:,k)=X(:,k)+T*zengguang(X(:,k),u(k)); 
miu1_next=0.6;
if  Xnext(1,k)>=miu1_next
    temp1=4;
elseif  Xnext(1,k)<=(-miu1_next)
    temp1=-4;
else
    temp1=0.5*log((Xnext(1,k)+deta1*miu1_next)./(deta2*miu1_next-Xnext(1,k)));
end
Znext(:,k)=[temp1;Xnext(2,k);Xnext(3,k);Xnext(4,k)];
end

for i=1:20
for k=1:ST
   kec_delta_fai(:,k)=Fai(Z(:,(k)))-Fai(Znext(:,(k)));
   kec_Q(k)=T*(Z(:,(k))'*Q*Z(:,(k))+Znext(:,(k))'*Q*Znext(:,(k)))*0.5;
   kec_lambd(:,k)=T*lambta*(Fai(Z(:,(k)))+Fai(Znext(:,(k))))*0.5;
   kec_u(k)=T*((Theta(Z(:,(k)))'*Wu(:,i))*R*(Theta(Z(:,(k)))'*Wu(:,i))+(Theta(Znext(:,(k)))'*Wu(:,i))*R*(Theta(Znext(:,(k)))'*Wu(:,i)))*0.5;
    kec_u_theta(k,:)=2*T*(((Theta(Z(:,(k)))'*Wu(:,i))-u((k)))'*Theta(Z(:,(k)))'+((Theta(Znext(:,(k)))'*Wu(:,i))-u((k)))'*Theta(Znext(:,(k)))')*0.5;
   AK(k,:)=[(kec_delta_fai(:,k)'+kec_lambd(:,k)'),  kec_u_theta(k,:)]; 
   GMAK(k)=kec_Q(k)+kec_u(k);
end
   AKsum=zeros(ST,32);
   GMAKsum=zeros(ST,1);
   for k=1:ST
   AKsum(k,:)=AK(k,:);
   GMAKsum(k,1)=GMAK(k);
   end  
    rank(AKsum'*AKsum)
   Wcu(i+1,:)=(AKsum'*AKsum)\AKsum'*GMAKsum;
   Wc(:,i+1)=Wcu(i+1,1:18)';
   V(i+1)=Fai(Z(:,(i+1)))'*Wc(:,i+1);
   Wu(:,i+1)=Wcu(i+1,19:32)';
    thgama(i)=( (AKsum*Wcu(i+1,:)'- GMAKsum)'*(AKsum*Wcu(i+1,:)'- GMAKsum));
end

figure
for iii=1:18
plot(Wc(iii,1:21))
hold on
end
ylabel('Critic Weights');
xlabel('The step of iterations');
legend('Weight 1','Weight 2','Weight 3','Weight 4','Weight 5','Weight 6','Weight 7','Weight 8','Weight 9','Weight 10','Weight 11','Weight 12','Weight 13','Weight 14','Weight 15','Weight 16','Weight 17','Weight 18');

figure
for iii=1:14
plot(Wu(iii,1:21))
hold on
end
ylabel('Actor Weights');
xlabel('The step of iterations');
legend('Weight 1','Weight 2','Weight 3','Weight 4','Weight 5','Weight 6','Weight 7','Weight 8','Weight 9','Weight 10','Weight 11','Weight 12','Weight 13','Weight 14');

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
