import torch
import copy
import numpy as np
from typing import Callable, List
import time
import argparse
import torch.nn.functional as Fn
from sklearn.utils.extmath import safe_sparse_dot
import matplotlib.pyplot as plt
import tls as ls
from matplotlib.font_manager import FontProperties
import matplotlib.lines as mlines




import psutil as psutil

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser(description='strongly_convex')
parser.add_argument('--seed', type=int, default=2)
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--x_loop', type=int, default=5000)
parser.add_argument('--y_loop', type=int, default=100)
parser.add_argument('--x_lr', type=float, default=0.1)
parser.add_argument('--y_lr', type=float, default=0.1)
parser.add_argument('--xSize', type=int, default=1000)
parser.add_argument('--ySize', type=int, default=1000)
parser.add_argument('--log', type=int, default=5)#10

args = parser.parse_args()

print(args)

def loss_L2(parameters):
    loss = 0
    for w in parameters:
        loss += torch.norm(w, 2) ** 2
    return loss

def positive_matrix(m):
    randt = torch.rand(m) + 1
    matrix0 = torch.diag(randt)
    invmatrix0 = torch.diag(1 / randt)
    Q = torch.rand(m, m)
    Q, R = torch.qr(Q)
    matrix = torch.mm(torch.mm(Q.t(), matrix0), Q)
    invmatrix = torch.mm(torch.mm(Q.t(), invmatrix0), Q)
    return matrix, invmatrix

#problem setting
A, invA = positive_matrix(args.ySize)
invA =torch.inverse(A)
z0 = torch.rand([args.xSize, 1]) * 1
D=torch.eye(args.xSize)
invaD= torch.inverse(invA + D)
xstar = torch.mm(invaD , z0)
ystar= torch.mm(invA , xstar)

def F(x, y):
    tmp = x - z0
    return 0.5 *torch.mm(tmp.t(),tmp) + 0.5 * torch.mm(y.t(),A@y)

def f(x, y):
    return 0.5  * torch.mm(y.t(),A@y) - torch.mm(x.t(), y )

#calculate gradient
def f_y(x,y, retain_graph=False, create_graph=False):
    loss = f(x,y)
    grad = torch.autograd.grad(loss, y,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def f_x(x,y, retain_graph=False, create_graph=False):
    loss = f(x,y)
    grad = torch.autograd.grad(loss, x,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def F_y(x,y, retain_graph=False, create_graph=False):
    loss = F(x,y)
    grad = torch.autograd.grad(loss, y,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def F_x(x,y, retain_graph=False, create_graph=False):
    loss = F(x,y)
    grad = torch.autograd.grad(loss, x,
                               retain_graph=retain_graph,
                               create_graph=create_graph)[0]
    return grad

def f_yy1(x,y):
    return A
    
def f_xy(x,y,vs):
    gra=torch.autograd.grad(f(x,y), y, retain_graph=True,allow_unused=True,create_graph=True,only_inputs=True)[0]
    gra.requires_grad_(True)
    grad=torch.autograd.grad(gra, x, grad_outputs=vs, retain_graph=True,
                                 allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(x)
      
def f_yy(x,y,vs):
    gra=torch.autograd.grad(f(x,y), y, retain_graph=True,allow_unused=True,create_graph=True,only_inputs=True)[0]
    gra.requires_grad_(True)
    grad=torch.autograd.grad(gra, y, grad_outputs=vs, retain_graph=True,
                                 allow_unused=True)[0]
    return grad if grad is not None else torch.zeros_like(y)

def f_y_yhat_x(y, yhat, x, retain_graph=False, create_graph=False):
    loss = f(x, y) - f(x,yhat.detach())
    grad = torch.autograd.grad(loss, [y, x],
                               retain_graph=retain_graph,
                               create_graph=create_graph)
    return loss, grad[0], grad[1]

#tools
def cg(A, b, x, num_steps):
    r = b - A @ x
    p = r.clone()
    rs_old = torch.dot(r.view(-1), r.view(-1))  # 将 r 转换为 1D 张量
    for _ in range(num_steps):
        Ap = A @ p
        alpha = rs_old / torch.dot(p.view(-1), Ap.view(-1))  # 将 p 和 Ap 转换为 1D 张量
        x = x + alpha * p
        r = r - alpha * Ap
        rs_new = torch.dot(r.view(-1), r.view(-1))  # 将 r 转换为 1D 张量
        if torch.sqrt(rs_new) < 1e-10:
            break
        p = r + (rs_new / rs_old) * p
        rs_old = rs_new
    return x

def bfgs(x,y,tol,step,maxiter_hg,m,h0,ex_up=False): 
            y_list, s_list, mu_list = [], [], []
            y1_list, s1_list, mu1_list = [], [], [] 
            for k in range(1, step + 1):
                if k<3:
                   s=-f_y(x,y)
                   y=y+0.01*s
                   new_grad=f_y(x,y)
                   ngrad=new_grad.detach().numpy()
                   ngrad=np.squeeze(ngrad)
                else:                
                   p = two_loops(grady, m, s_list, y_list, mu_list,h0)
                   s= p
                   s=np.expand_dims(s,axis=1)
                   st=torch.from_numpy(s)
                   y=y+st
                   new_grad=f_y(x,y)#
                   ngrad=new_grad.detach().numpy()#\nabla_y f(x_k,y_{k+1})
                   ngrad=np.squeeze(ngrad)
                   yg=ngrad-grady
                   yg=np.squeeze(yg)
                   s=np.squeeze(s)
                    # Update the memory
                   if (safe_sparse_dot(yg,s))>1e-5:
                       y_list.append(yg.copy())
                       s_list.append(s.copy())
                       mu=1/safe_sparse_dot(yg,s)
                       mu_list.append(mu)
                   if len(y_list) > m:
                      y_list.pop(0)
                      s_list.pop(0)
                      mu_list.pop(0)
                grady=ngrad
            ogrady = F_y(x,y)# dy F
            gradFy=ogrady.detach().numpy()#\nabla_y F(x_k,y_{k+1})
            gradFy=np.squeeze(gradFy)
            
            if ex_up==False:
               hg = -two_loops(gradFy, m, s_list, y_list, mu_list,h0)
               hg=np.expand_dims(hg,axis=1)
               et=torch.from_numpy(hg)
            else:
                for i in range (1, maxiter_hg + 1):
                    eq = -two_loops(gradFy, m, s1_list, y1_list, mu1_list,h0)#default H0=I
                    eq=np.expand_dims(eq,axis=1)
                    et=torch.from_numpy(eq)
                    f1grad=f_y(x,y+et)
                    f1grad=f1grad.detach().numpy()
                    f1grad=np.squeeze(f1grad)
                    eq=np.squeeze(eq)
                    y_tilde1 = f1grad- grady
                    if safe_sparse_dot(y_tilde1, eq)>1e-10:
                       mu1 = 1 / safe_sparse_dot(y_tilde1, eq)
                       y1_list.append(y_tilde1.copy())
                       s1_list.append(eq.copy())
                       mu1_list.append(mu1)
                    if len(y1_list) > m:
                       y1_list.pop(0)
                       s1_list.pop(0)
                       mu1_list.pop(0)
            
            print(f'{k} iterates')
            return y, et

def two_loops(grad_y, m, s_list, y_list, mu_list,h0):
            q = grad_y.copy()
            alpha_list = []
            for s, y, mu in zip(reversed(s_list), reversed(y_list), reversed(mu_list)):
                alpha = mu * safe_sparse_dot(s, q)
                alpha_list.append(alpha)
                q -= alpha * y
            r=q
            
            for s, y, mu, alpha in zip(s_list, y_list, mu_list, reversed(alpha_list)):
                beta = mu * safe_sparse_dot(y, r)
                r += (alpha - beta) * s
            return -r

def rbfgs(x,y,tol,step,m,h0,exup=False): 
            c1=0.0001
            c2=0.0009
            y_list, s_list, mu_list = [], [], []
            new_grad=f_y(x,y)
            grady=new_grad.detach().numpy()
            grady=np.squeeze(grady)
            lf=lambda y: f(x,y)
            lf_grad=lambda y:f_y(x,y)
            t=0.01
            tc=1e-9
            obf=f(x,y)
            maxls=10
            for k in range(1, step + 1):
                   if k>1 and k%5==0 and exup==True:
                          ogrady = F_y(x,y)# dy F
                          gradFy=ogrady.detach().numpy()#\nabla_y F(x_k,y_{k+1})
                          gradFy=np.squeeze(gradFy)
                          eqo = -two_loops(gradFy, m, s_list, y_list, mu_list,h0)
                          eqo=eqo/np.linalg.norm(eqo)*np.linalg.norm(s)
                          eq1=np.expand_dims(eqo,axis=1)
                          eq1=torch.from_numpy(eq1)
                          fy=f_y(x,y+eq1)
                          fy=fy.detach().numpy()
                          fy=np.squeeze(fy)
                          y_tildeo=fy-grady
                          if safe_sparse_dot(y_tildeo, eqo)>1e-10:
                            mu = 1 / safe_sparse_dot(y_tildeo, eqo)
                            y_list.append(y_tildeo.copy())
                            s_list.append(eqo.copy())
                            mu_list.append(mu)                       
                   d = two_loops(grady, m, s_list, y_list, mu_list,h0)
                   p=np.expand_dims(d,axis=1)
                   p=torch.from_numpy(p)
                   gtd=(new_grad.view(-1)).dot(p.view(-1))
                   obf, new_grad, step,lsi = ls.strong_wolfe(lf, lf_grad, y,t,
                                                              p, obf,new_grad,gtd,
                                                              c1,c2,tc,
                                                              maxls)
                   if step is None:
                        step = 0.01
                        s = step * d
                        s=np.expand_dims(s,axis=1)
                        st=torch.from_numpy(s)
                        y = y +st
                        new_grad=f_y(x,y)
                   else: 
                        if type(step)!=float:
                            if type(step)!=int:
                               step =step.detach().numpy()[0]
                        s = step * d
                        s=np.expand_dims(s,axis=1)
                        st=torch.from_numpy(s)
                        y = y +st
                   ngrad=new_grad.detach().numpy()
                   ngrad=np.squeeze(ngrad)
                   yg=ngrad-grady
                   yg=np.squeeze(yg)
                   s=np.squeeze(s)
                    # Update the memory
                   if (safe_sparse_dot(yg,s))>1e-10:
                       y_list.append(yg.copy())
                       s_list.append(s.copy())
                       mu=1/safe_sparse_dot(yg,s)
                       mu_list.append(mu)
                   if len(y_list) > m:
                      y_list.pop(0)
                      s_list.pop(0)
                      mu_list.pop(0)
                   grady=ngrad
                   l_inf_norm_grad = np.linalg.norm(grady)
                   if l_inf_norm_grad < tol:
                      break
            
            ogrady = F_y(x,y)
            gradFy=ogrady.detach().numpy()
            gradFy=np.squeeze(gradFy)
            
            hg = -two_loops(gradFy, m, s_list, y_list, mu_list,h0)
            hg=np.expand_dims(hg,axis=1)
            et=torch.from_numpy(hg)
             
            print(f'{k} iterates')
            return y, et

def sr(x,y,tol,step,maxiter_hg,m,h0,ex_up=False): 
            y_list, s_list, mu_list = [], [], []
            y1_list, s1_list, mu1_list = [], [], [] 
            for k in range(1, step + 1):
                if k<10:
                   s=-f_y(x,y)
                   y=y+0.01*s
                   new_grad=f_y(x,y)
                   ngrad=new_grad.detach().numpy()
                   ngrad=np.squeeze(ngrad)
                else:
                   p = two_loopsr(grady, m, s_list, y_list,h0)
                   s= -p
                   s=np.expand_dims(s,axis=1)
                   st=torch.from_numpy(s)
                   y=y+st
                   new_grad=f_y(x,y)#[0]
                   ngrad=new_grad.detach().numpy()
                   ngrad=np.squeeze(ngrad)
                   yg=ngrad-grady
                   yg=np.squeeze(yg)
                   s=np.squeeze(s)
                   if (safe_sparse_dot(yg,s))>1e-5:
                       y_list.append(yg.copy())
                       s_list.append(s.copy())
                   if len(y_list) > m:
                      y_list.pop(0)
                      s_list.pop(0)
                grady=ngrad
                
            ogrady = F_y(x,y)
            gradFy=ogrady.detach().numpy()
            gradFy=np.squeeze(gradFy)
            
            if ex_up==False:
               hg = two_loopsr(gradFy, m, s_list, y_list,h0)
               hg=np.expand_dims(hg,axis=1)
               et=torch.from_numpy(hg)
            else:
                for i in range (1, maxiter_hg + 1):
                    eq = two_loopsr(gradFy, m, s1_list, y1_list,h0)
                    eq=np.expand_dims(eq,axis=1)
                    et=torch.from_numpy(eq)
                    f1grad=f_y(x,y+et)
                    f1grad=f1grad.detach().numpy()
                    f1grad=np.squeeze(f1grad)
                    eq=np.squeeze(eq)
                    y_tilde1 = f1grad- grady
                    if safe_sparse_dot(y_tilde1, eq)>1e-6:
                       y1_list.append(y_tilde1.copy())
                       s1_list.append(eq.copy())
                    if len(y1_list) > m:
                       y1_list.pop(0)
                       s1_list.pop(0)
            
            print(f'{k} iterates')
            return y, et

def two_loopsr(grad_x, m,s_list, y_list,h0):
            q = grad_x.copy()
            p_list = []
            r=q
            for s, y in zip(s_list, y_list):
                p=s-y     #p_i=s_i-H0y_i
                i=len(p_list)
                for k in range(i):
                    p = p-(safe_sparse_dot(p_list[k], y))/(safe_sparse_dot(p_list[k], y_list[k]))*p_list[k]
                p_list.append(p)
            for p, y in zip(p_list, y_list):
                r = r+(safe_sparse_dot(p,q))/(safe_sparse_dot(p, y))*p
            return r

def TN(A: torch.Tensor, b: torch.Tensor, K: int = 1, inner_lr: float = 0.1) -> torch.Tensor:   
    p = v = b.clone()
    for _ in range(K):
        output = torch.matmul(A, v)
        v = v - inner_lr * output
        p = v + p
        
        if torch.norm(v) < 1e-6:
            break
    
    return inner_lr *p


#Initialization  
x0=1
y0=1
tt=3
x = (float(x0) * torch.ones([args.xSize,1])).requires_grad_(True)
y = (float(y0) * torch.ones([args.ySize,1])).requires_grad_(True)
x_loop = args.x_loop

yhat= copy.deepcopy(y)
       
with torch.no_grad():
    xgard0=torch.mm(D+invA,x)-z0
    dx0=torch.norm(xgard0)
    xdis0=torch.norm(x - xstar) /torch.norm( xstar)
    ydis0=torch.norm(y-ystar) / torch.norm(ystar)
    print(dx0)
    print(xdis0)
u1=1e-4
xgrad=torch.zeros([args.xSize, 1])

#f2sa
xdislistf2sa=[]
ydislistf2sa=[]
dxlistf2sa=[]
timelistf2sa= [] 


inner_opt = torch.optim.SGD([
        {'params': [y], 'lr': args.y_lr},
        {'params': [yhat], 'lr': args.y_lr}])
outer_opt = torch.optim.SGD([x], lr=args.x_lr)


total_time = 0.0
timelistf2sa.append(total_time)
xdislistf2sa.append(copy.deepcopy(xdis0.detach().cpu().numpy()))
ydislistf2sa.append(copy.deepcopy(ydis0.detach().cpu().numpy()))
dxlistf2sa.append(copy.deepcopy(dx0.detach().cpu().numpy()))

lmbd=0.1

for x_itr in range(x_loop):
            t0 = time.time()
            yhat.data = y.data.clone()#
            for it in range(10):
               Fy = F_y(x, y)
               fy= f_y(x,y)
               inner_opt.zero_grad()
               y.grad=F_y(x,y) + lmbd * fy
               yhat.grad=f_y(x,yhat)
               inner_opt.step()

            # prepare gradients 
               
            # prepare gradients 
            fx_minus_fx_yk=torch.zeros_like(x)
            _,_, fx_minus_fx_yk = f_y_yhat_x(y, yhat, x)

            outer_opt.zero_grad()
            Fx=F_x(x,y)
            x.grad = Fx+lmbd * fx_minus_fx_yk
            outer_opt.step()
            lmbd=lmbd+0.001
            t1 = time.time()
            total_time += t1 - t0
            
            xgrad.data=x.grad.data.clone()
          
            with torch.no_grad():
              xgard=torch.mm(D+invA,x)-z0
              dx=torch.norm(xgrad-xgard)
              xdis=torch.norm(x - xstar) /torch.norm( xstar)
              ydis=torch.norm(y-ystar) / torch.norm(ystar)

    
            if x_itr % args.log == 0:
                print('x_itr={},xdist={:.6f},ydist={:.6f}, total_time={:.6f}'.format(
                x_itr,  xdis.detach().cpu().numpy(),ydis.detach().cpu().numpy(), total_time))
                print(torch.norm(xgrad))
                """
                print(torch.norm(xgard))
                print(dx)
                """
            timelistf2sa.append(total_time)
            dxlistf2sa.append(copy.deepcopy(dx.detach().cpu().numpy()))
            xdislistf2sa.append(copy.deepcopy(xdis.detach().cpu().numpy()))
            ydislistf2sa.append(copy.deepcopy(ydis.detach().cpu().numpy()))
            if total_time>tt:
                    break
state_dict = {'time': timelistf2sa,
                          'xdist':xdislistf2sa,
                          'ydist':ydislistf2sa
                          }
torch.save(state_dict, 'f2sa.pt')      



#bome
xdislistbme=[]
ydislistbme=[]
#dxlistbme=[]
timelistbme= [] 
x = (float(x0) * torch.ones([args.xSize,1])).requires_grad_(True)
y = (float(y0) * torch.ones([args.ySize,1])).requires_grad_(True)

#dxlistbme.append(copy.deepcopy(dx0.detach().cpu().numpy()))
outer_opt = torch.optim.SGD(
             [
             {'params': [y], 'lr': args.y_lr},
             {'params': [x], 'lr': args.x_lr},
             ])
inner_opt = torch.optim.SGD([yhat], lr=args.y_lr)

total_time = 0.0
timelistbme.append(total_time)
xdislistbme.append(copy.deepcopy(xdis0.detach().cpu().numpy()))
ydislistbme.append(copy.deepcopy(ydis0.detach().cpu().numpy()))

for x_itr in range(x_loop):
            t0 = time.time()
            yhat.data = y.data.clone()#
            for it in range(1):
               inner_opt.zero_grad()
               yhat.grad=f_y(x,yhat)
               inner_opt.step()

            # prepare gradients 
            Fy=F_y(x,y)
            Fx=F_x(x,y)
            loss, fy, fx_minus_fx_yk = f_y_yhat_x(y, yhat, x)

            dF=torch.cat([Fy.view(-1), Fx.view(-1)])
            df= torch.cat([fy.view(-1), fx_minus_fx_yk.view(-1)])
            norm_dq = df.norm().pow(2)
            dot = dF.dot(df)
            lmbd = Fn.relu((u1 * loss - dot)/(norm_dq + 1e-8))

            outer_opt.zero_grad()
            y.grad = Fy + lmbd * fy
            x.grad = Fx+lmbd * fx_minus_fx_yk
            outer_opt.step()
            t1 = time.time()
            total_time += t1 - t0
            xgrad.data=x.grad.data.clone()
          
            with torch.no_grad():
              xdis=torch.norm(x - xstar) /torch.norm( xstar)
              ydis=torch.norm(y-ystar) / torch.norm(ystar)

    
            if x_itr % args.log == 0:
                print('x_itr={},xdist={:.6f},ydist={:.6f}, total_time={:.6f}'.format(
                x_itr,  xdis.detach().cpu().numpy(),ydis.detach().cpu().numpy(), total_time))
                print(torch.norm(xgrad))
            timelistbme.append(total_time)
            xdislistbme.append(copy.deepcopy(xdis.detach().cpu().numpy()))
            ydislistbme.append(copy.deepcopy(ydis.detach().cpu().numpy()))
            if total_time>tt:
                    break
state_dict = {'time': timelistbme,
                          'xdist':xdislistbme,
                          'ydist':ydislistbme
                          }
torch.save(state_dict, 'bome.pt')      

#shinea
xdislistshinea=[]
ydislistshinea=[]
dxlistshinea=[]
timelistshinea = [] 
x = (float(x0) * torch.ones([args.xSize, 1])).requires_grad_(True)
y = (float(y0) * torch.ones([args.ySize, 1])).requires_grad_(True)

total_time = 0.0
dxlistshinea.append(copy.deepcopy(dx0.detach().cpu().numpy()))
timelistshinea.append(total_time)
xdislistshinea.append(copy.deepcopy(xdis0.detach().cpu().numpy()))
ydislistshinea.append(copy.deepcopy(ydis0.detach().cpu().numpy()))

for x_itr in range(x_loop):
    t0 = time.time()
    y,et= rbfgs(x,y,tol=1/(1*(x_itr+1)),step=100,m=30,h0=0.1,exup=True)
    
    Fx=F_x(x,y)
    xgrad=Fx+et

    x=x-0.1*xgrad
    t1 = time.time()
    total_time += t1 - t0


    with torch.no_grad():
              xgard=torch.mm(D+invA,x)-z0
              dx=torch.norm(xgrad-xgard)
              xdis=torch.norm(x - xstar) /torch.norm( xstar)
              ydis=torch.norm(y-ystar) / torch.norm(ystar)

    
    if x_itr % args.log == 0:
                print('x_itr={},xdist={:.6f},ydist={:.6f}, total_time={:.6f}'.format(
                x_itr,  xdis.detach().cpu().numpy(),ydis.detach().cpu().numpy(), total_time))
                print(torch.norm(xgrad))
                print(torch.norm(xgard))
                print(dx)
    
    timelistshinea.append(total_time)
    dxlistshinea.append(copy.deepcopy(dx.detach().cpu().numpy()))
    xdislistshinea.append(copy.deepcopy(xdis.detach().cpu().numpy()))
    ydislistshinea.append(copy.deepcopy(ydis.detach().cpu().numpy()))
    if total_time>tt:
        break

state_dict = {'time': timelistshinea,
                          'dx': dxlistshinea,
                          'xdist':xdislistshinea,
                          'ydist':ydislistshinea
                          }
torch.save(state_dict, 'shine1.pt')    


#qnbo
xdislistfoae=[]
ydislistfoae=[]
dxlistfoae=[]
timelistfoae= [] 
x = (float(x0) * torch.ones([args.xSize, 1])).requires_grad_(True)
y = (float(y0) * torch.ones([args.ySize, 1])).requires_grad_(True)
dxlistfoae.append(copy.deepcopy(dx0.detach().cpu().numpy()))

total_time = 0.0
timelistfoae.append(total_time)
xdislistfoae.append(copy.deepcopy(xdis0.detach().cpu().numpy()))
ydislistfoae.append(copy.deepcopy(ydis0.detach().cpu().numpy()))

for x_itr in range(x_loop):
    t0 = time.time()
    y,et= bfgs(x,y,tol=1/(100*(x_itr+1)),step=15,maxiter_hg=x_itr+1,m=30,h0=0.1,ex_up=True)
    
    Fx=F_x(x,y)
    xgrad=Fx+et

    x=x-0.1*xgrad
    t1 = time.time()
    total_time += t1 - t0


    with torch.no_grad():
              xgard=torch.mm(D+invA,x)-z0
              dx=torch.norm(xgrad-xgard)
              xdis=torch.norm(x - xstar) /torch.norm( xstar)
              ydis=torch.norm(y-ystar) / torch.norm(ystar)

    
    if x_itr % args.log == 0:
                print('x_itr={},xdist={:.6f},ydist={:.6f}, total_time={:.6f}'.format(
                x_itr,  xdis.detach().cpu().numpy(),ydis.detach().cpu().numpy(), total_time))
                print(torch.norm(xgrad))
                print(torch.norm(xgard))
                print(dx)
    
    timelistfoae.append(total_time)
    dxlistfoae.append(copy.deepcopy(dx.detach().cpu().numpy()))
    xdislistfoae.append(copy.deepcopy(xdis.detach().cpu().numpy()))
    ydislistfoae.append(copy.deepcopy(ydis.detach().cpu().numpy()))
    if total_time>tt:
        break
state_dict = {'time': timelistfoae,
                          'dx': dxlistfoae,
                          'xdist':xdislistfoae,
                          'ydist':ydislistfoae
                          }
torch.save(state_dict, 'foa1.pt')  

#sr
xdislistsr=[]
ydislistsr=[]
dxlistsr=[]
timelistsr = [] 
x = (float(x0) * torch.ones([args.xSize, 1])).requires_grad_(True)
y = (float(y0) * torch.ones([args.ySize, 1])).requires_grad_(True)

total_time = 0.0
dxlistsr.append(copy.deepcopy(dx0.detach().cpu().numpy()))
timelistsr.append(total_time)
xdislistsr.append(copy.deepcopy(xdis0.detach().cpu().numpy()))
ydislistsr.append(copy.deepcopy(ydis0.detach().cpu().numpy()))

for x_itr in range(x_loop):
    t0 = time.time()
    y,et= sr(x,y,tol=1/(100*(x_itr+1)),step=15,maxiter_hg=25,m=30,h0=1,ex_up=True)
    
    Fx=F_x(x,y)
    xgrad=Fx+et

    x=x-0.5*xgrad
    t1 = time.time()
    total_time += t1 - t0


    with torch.no_grad():
              xgard=torch.mm(D+invA,x)-z0
              dx=torch.norm(xgrad-xgard)
              xdis=torch.norm(x - xstar) /torch.norm( xstar)
              ydis=torch.norm(y-ystar) / torch.norm(ystar)

    
    if x_itr % args.log == 0:
                print('x_itr={},xdist={:.6f},ydist={:.6f}, total_time={:.6f}'.format(
                x_itr,  xdis.detach().cpu().numpy(),ydis.detach().cpu().numpy(), total_time))
                print(torch.norm(xgrad))
                print(torch.norm(xgard))
                print(dx)
    
    timelistsr.append(total_time)
    dxlistsr.append(copy.deepcopy(dx.detach().cpu().numpy()))
    xdislistsr.append(copy.deepcopy(xdis.detach().cpu().numpy()))
    ydislistsr.append(copy.deepcopy(ydis.detach().cpu().numpy()))
    if total_time>tt:
        break
state_dict = {'time': timelistsr,
                          'dx': dxlistsr,
                          'xdist':xdislistsr,
                          'ydist':ydislistsr
                          }
torch.save(state_dict, 'sr.pt')  

#AID-BIO-TN
xdislistaid=[]
ydislistaid=[]
dxlistaid=[]
timelistaid= []
alpha = 0.01
beta = 0.2
T=1
x = (float(x0) * torch.ones([args.xSize,1])).requires_grad_(True)
y = (float(y0) * torch.ones([args.ySize,1])).requires_grad_(True)
v = torch.zeros_like(y)

dxlistaid.append(copy.deepcopy(dx0.detach().cpu().numpy()))

total_time = 0.0
timelistaid.append(total_time)
xdislistaid.append(copy.deepcopy(xdis0.detach().cpu().numpy()))
ydislistaid.append(copy.deepcopy(ydis0.detach().cpu().numpy()))

for x_itr in range(x_loop):
        t0 = time.time()
        yhat.data = y.data.clone()#
        for it in range(T):
             y= y - alpha * f_y(x,y)    
        # prepare gradients 
        Fy=F_y(x,y)
        Fx=F_x(x,y)
        fyy=f_yy1(x,y)
        v = TN(fyy,Fy)
        #fyx=f_xy(x,y,v)
        
        grad_Phi = Fx+v

        x = x - beta * grad_Phi

        t1 = time.time()
        total_time += t1 - t0
        #xgrad.data=x.grad.data.clone()
        
        with torch.no_grad():
            xgard=torch.mm(D+invA,x)-z0
            dx=torch.norm(grad_Phi-xgard)
            xdis=torch.norm(x - xstar) /torch.norm(xstar)
            ydis=torch.norm(y-ystar) / torch.norm(ystar)


        if x_itr % args.log == 0:
            print('x_itr={},xdist={:.6f},ydist={:.6f}, total_time={:.6f}'.format(
            x_itr,  xdis.detach().cpu().numpy(),ydis.detach().cpu().numpy(), total_time))
            print(torch.norm(xgrad))
        timelistaid.append(total_time)
        dxlistaid.append(copy.deepcopy(dx.detach().cpu().numpy()))
        xdislistaid.append(copy.deepcopy(xdis.detach().cpu().numpy()))
        ydislistaid.append(copy.deepcopy(ydis.detach().cpu().numpy()))
        if total_time>tt:
                break
state_dict = {'time': timelistaid,
                        'dx': dxlistaid,
                        'xdist':xdislistaid,
                        'ydist':ydislistaid
                        }
torch.save(state_dict, 'AID-BIO.pt')


#AID-BIO-CG
xdislistbio=[]
ydislistbio=[]
dxlistbio=[]
timelistbio= []
alpha = 0.01
beta = 0.2
T=1
x = (float(x0) * torch.ones([args.xSize,1])).requires_grad_(True)
y = (float(y0) * torch.ones([args.ySize,1])).requires_grad_(True)
v = torch.zeros_like(y)

dxlistbio.append(copy.deepcopy(dx0.detach().cpu().numpy()))

total_time = 0.0
timelistbio.append(total_time)
xdislistbio.append(copy.deepcopy(xdis0.detach().cpu().numpy()))
ydislistbio.append(copy.deepcopy(ydis0.detach().cpu().numpy()))

for x_itr in range(x_loop):
        t0 = time.time()
        yhat.data = y.data.clone()#
        for it in range(T):
             y= y - alpha * f_y(x,y)    
        # prepare gradients 
        Fy=F_y(x,y)
        Fx=F_x(x,y)
        fyy=f_yy1(x,y)
        v = cg(fyy,Fy,v,1)
        #fyx=f_xy(x,y,v)
        
       
        grad_Phi = Fx+v
        x = x - beta * grad_Phi

        t1 = time.time()
        total_time += t1 - t0
        #xgrad.data=x.grad.data.clone()
        
        with torch.no_grad():
            xgard=torch.mm(D+invA,x)-z0
            dx=torch.norm(grad_Phi-xgard)
            xdis=torch.norm(x - xstar) /torch.norm( xstar)
            ydis=torch.norm(y-ystar) / torch.norm(ystar)


        if x_itr % args.log == 0:
            print('x_itr={},xdist={:.6f},ydist={:.6f}, total_time={:.6f}'.format(
            x_itr,  xdis.detach().cpu().numpy(),ydis.detach().cpu().numpy(), total_time))
            print(torch.norm(xgrad))
        timelistbio.append(total_time)
        dxlistbio.append(copy.deepcopy(dx.detach().cpu().numpy()))
        xdislistbio.append(copy.deepcopy(xdis.detach().cpu().numpy()))
        ydislistbio.append(copy.deepcopy(ydis.detach().cpu().numpy()))
        if total_time>tt:
                break
state_dict = {'time': timelistbio,
                        'dx': dxlistbio,
                        'xdist':xdislistbio,
                        'ydist':ydislistbio
                        }
torch.save(state_dict, 'AID-BIO.pt')

#AMIGO-CG
xdislistami=[]
ydislistami=[]
dxlistami=[]
timelistami= [] 
gamma = 0.2
alpha = 0.01
T=1
x = (float(x0) * torch.ones([args.xSize,1])).requires_grad_(True)
y = (float(y0) * torch.ones([args.ySize,1])).requires_grad_(True)
z = torch.zeros_like(y)
total_time = 0.0
dxlistami.append(copy.deepcopy(dx0.detach().cpu().numpy()))
timelistami.append(total_time)
xdislistami.append(copy.deepcopy(xdis0.detach().cpu().numpy()))
ydislistami.append(copy.deepcopy(ydis0.detach().cpu().numpy()))
for x_itr in range(x_loop):
            t0 = time.time()
            for it in range(T):
                y = y - alpha * f_y(x,y)
            # prepare gradients 
            Fy=F_y(x,y)
            Fx=F_x(x,y)
            fyy=f_yy1(x,y)
            z=cg(fyy,Fy,z,1)
            #w=f_xy(x,y,z)
            x_grad=Fx+z
            x=x-gamma*x_grad

            t1 = time.time()
            total_time += t1 - t0
          
            with torch.no_grad():
                  xgard=torch.mm(D+invA,x)-z0
                  dx=torch.norm(x_grad-xgard)
                  xdis=torch.norm(x - xstar) /torch.norm( xstar)
                  ydis=torch.norm(y-ystar) / torch.norm(ystar)

    
            if x_itr % args.log == 0:
                print('x_itr={},xdist={:.6f},ydist={:.6f}, total_time={:.6f}'.format(
                x_itr,  xdis.detach().cpu().numpy(),ydis.detach().cpu().numpy(), total_time))
                print(torch.norm(xgrad))
            timelistami.append(total_time)
            dxlistami.append(copy.deepcopy(dx.detach().cpu().numpy()))
            xdislistami.append(copy.deepcopy(xdis.detach().cpu().numpy()))
            ydislistami.append(copy.deepcopy(ydis.detach().cpu().numpy()))
            if total_time>tt:
                    break
state_dict = {'time': timelistami,
                          'dx': dxlistami,
                          'xdist':xdislistami,
                          'ydist':ydislistami
                          }
torch.save(state_dict, 'AMIGO-CG.pt')      

#PZOBO
xdislistpzo=[]
ydislistpzo=[]
dxlistpzo=[]
timelistpzo= [] 
alpha = 0.01
beta = 0.01
mu = 100
Q=10
N=10
x = (float(x0) * torch.ones([args.xSize,1])).requires_grad_(True)
y = (float(y0) * torch.ones([args.ySize,1])).requires_grad_(True)
y_kj_t = y.clone()
sum_term = torch.zeros_like(y)
total_time = 0.0
dxlistpzo.append(copy.deepcopy(dx0.detach().cpu().numpy()))
timelistpzo.append(total_time)
xdislistpzo.append(copy.deepcopy(xdis0.detach().cpu().numpy()))
ydislistpzo.append(copy.deepcopy(ydis0.detach().cpu().numpy()))
for x_itr in range(x_loop):
            t0 = time.time()
            yhat.data = y.data.clone()#
            for it in range(N):
                y= y - alpha * f_y(x,y,retain_graph=True)
            # second step
            for j in range(Q):
                u_kj = torch.randn_like(x)  
                for t in range(1, N + 1):
                    # prepare gradients
                    gy=f_y(x+mu*u_kj,y_kj_t,retain_graph=True)
                    y_kj_t = y_kj_t - alpha * gy
                delta_j = (y_kj_t - y) / mu
                fy=f_y(x,y)
                inner = (delta_j.T @ fy)
                sum_term += inner * u_kj
            fx=f_x(x,y)
            grad_phi = fx + (1/Q)*sum_term
            x = x - beta * grad_phi
            t1 = time.time()
            total_time += t1 - t0
            with torch.no_grad():
                  xgard=torch.mm(D+invA,x)-z0
                  dx=torch.norm(grad_Phi-xgard)
                  xdis=torch.norm(x - xstar) /torch.norm( xstar)
                  ydis=torch.norm(y-ystar) / torch.norm(ystar)

    
            if x_itr % args.log == 0:
                print('x_itr={},xdist={:.6f},ydist={:.6f}, total_time={:.6f}'.format(
                x_itr,  xdis.detach().cpu().numpy(),ydis.detach().cpu().numpy(), total_time))
                print(torch.norm(xgrad))
            timelistpzo.append(total_time)
            dxlistpzo.append(copy.deepcopy(dx.detach().cpu().numpy()))
            xdislistpzo.append(copy.deepcopy(xdis.detach().cpu().numpy()))
            ydislistpzo.append(copy.deepcopy(ydis.detach().cpu().numpy()))
            if total_time>tt:
                    break
state_dict = {'time': timelistpzo,
                          'dx': dxlistpzo,
                          'xdist':xdislistpzo,
                          'ydist':ydislistpzo
                          }
torch.save(state_dict, 'PZOBO.pt')    
         
lw = 2


styles = {
    'qNBO(SR1)': {'color': '#0000CC', 'linestyle': '-', 'linewidth': 6},
    'qNBO(BFGS)': {'color': '#D02020', 'linestyle': '-', 'linewidth': 6}, 
    "BOME": {'color': 'C9', 'linestyle': '-', 'linewidth': 6, 'alpha': 0.6},
    'F2SA': {'color': 'C0', 'linestyle': '--', 'linewidth': 5 },
    "SHINE-OPA": {'color': 'green', 'linestyle': '-', 'linewidth': 6},
    "AID-TN": {'color': 'orange', 'linestyle': '-', 'linewidth': 6},
    "AID-BIO": {'color': 'purple', 'linestyle': '-', 'linewidth': 6},
    "AMIGO-CG": {'color': 'pink', 'linestyle': '-', 'linewidth': 6},
    "PZOBO": {'color': 'yellow', 'linestyle': '-', 'linewidth': 6},
}


plt.figure(figsize=(10, 9))
ticks = [x * 0.5 for x in range(0, 6)]  # This will create [0, 0.5, 1.0, 1.5]


plt.plot(timelistfoae, xdislistfoae, label='qNBO(BFGS)', **styles["qNBO(BFGS)"])
plt.plot(timelistsr, xdislistsr, label='qNBO(SR1)', **styles["qNBO(SR1)"])
plt.plot(timelistf2sa, xdislistf2sa, label='F2SA', **styles["F2SA"])
plt.plot(timelistshinea, xdislistshinea, label='SHINE-OPA', **styles["SHINE-OPA"])
plt.plot(timelistbme, xdislistbme,  label='BOME', **styles["BOME"])
plt.plot(timelistaid, xdislistaid,  label='AID-TN', **styles["AID-TN"])
plt.plot(timelistbio, xdislistbio,  label='AID-BIO', **styles["AID-BIO"])
plt.plot(timelistami, xdislistami,  label='AMIGO-CG', **styles["AMIGO-CG"])
plt.plot(timelistpzo, xdislistpzo,  label='PZOBO', **styles["PZOBO"])

plt.xlabel('Running time (s)', fontsize=30)
plt.ylabel(r'$||x-x*||/||x*||$', fontsize=30, fontweight=900)
plt.xticks(fontsize=20)
plt.xticks(ticks)
plt.legend(fontsize=27,loc='center right', bbox_to_anchor=(1, 0.35))
plt.xlim(-0.05,3)
plt.yticks(fontsize=20)
plt.yscale('log')
plt.grid(visible=True, which='major', linestyle='-.', alpha=0.7)

plt.savefig('xdisn.pdf', dpi=300, bbox_inches='tight')



plt.figure(figsize=(10, 9))

ticks = [x * 0.5 for x in range(0, 6)]  
plt.plot(timelistf2sa, dxlistf2sa, label='F2SA', **styles["F2SA"])
plt.plot(timelistsr, dxlistsr, label='qNBO(SR1)', **styles["qNBO(SR1)"])
plt.plot(timelistfoae, dxlistfoae, label='qNBO(BFGS)', **styles["qNBO(BFGS)"])
plt.plot(timelistshinea, dxlistshinea, label='SHINE-OPA', **styles["SHINE-OPA"])
plt.plot(timelistshinea, dxlistshinea, label='SHINE-OPA', **styles["SHINE-OPA"])
plt.plot(timelistaid, dxlistaid, label='AID-TN', **styles["AID-TN"])
plt.plot(timelistbio, dxlistbio, label='AID-BIO', **styles["AID-BIO"])
plt.plot(timelistami, dxlistami, label='AMIGO-CG', **styles["AMIGO-CG"])
plt.plot(timelistpzo, dxlistpzo, label='PZOBO', **styles["PZOBO"])
plt.xlabel('Running time (s)', fontsize=30)
plt.ylabel(r'$||d_x-\nabla\Phi||$', fontsize=30, fontweight=900)
plt.xticks(fontsize=20)
plt.xticks(ticks)
plt.xlim(-0.05,3)

plt.yticks(fontsize=20)
plt.yscale('log')

plt.grid(visible=True, which='major', linestyle='-.', alpha=0.7)

plt.savefig('dx.pdf', dpi=300, bbox_inches='tight')


plt.figure(figsize=(10, 9))
plt.plot(timelistf2sa, xdislistf2sa, label='F2SA', **styles["F2SA"])
plt.plot(timelistsr, xdislistsr, label='qNBO(SR1)', **styles["qNBO(SR1)"])
plt.plot(timelistfoae, xdislistfoae, label='qNBO(BFGS)', **styles["qNBO(BFGS)"])
plt.plot(timelistshinea, xdislistshinea, label='SHINE-OPA', **styles["SHINE-OPA"])
plt.plot(timelistbme, xdislistbme,  label='BOME', **styles["BOME"])
plt.plot(timelistaid, xdislistaid,  label='AID-TN', **styles["AID-TN"])
plt.plot(timelistbio, xdislistbio,  label='AID-BIO', **styles["AID-BIO"])
plt.plot(timelistami, xdislistami,  label='AMIGO-CG', **styles["AMIGO-CG"])
plt.plot(timelistpzo, xdislistpzo,  label='PZOBO', **styles["PZOBO"])

plt.xlabel('Running time (s)', fontsize=30)
plt.ylabel(r'$||x-x*||/||x*||$', fontsize=30, fontweight=900)
plt.xticks(fontsize=20)
plt.xticks(ticks)
plt.xlim(-0.05,3)
plt.yticks(fontsize=20)
plt.yscale('log')
plt.grid(visible=True, which='major', linestyle='-.', alpha=0.7)

plt.savefig('xdis.pdf', dpi=300, bbox_inches='tight')

plt.figure(figsize=(10, 9))
plt.plot(timelistf2sa, ydislistf2sa, label='F2SA', **styles["F2SA"])
plt.plot(timelistsr, ydislistsr, label='qNBO(SR1)', **styles["qNBO(SR1)"])
plt.plot(timelistfoae, ydislistfoae, label='qNBO(BFGS)', **styles["qNBO(BFGS)"])
plt.plot(timelistshinea, ydislistshinea, label='SHINE-OPA', **styles["SHINE-OPA"])
plt.plot(timelistbme, ydislistbme, label='BOME', **styles["BOME"])
plt.plot(timelistaid, ydislistaid,  label='AID-TN', **styles["AID-TN"])
plt.plot(timelistbio, ydislistbio,  label='AID-BIO', **styles["AID-BIO"])
plt.plot(timelistami, ydislistami,  label='AMIGO-CG', **styles["AMIGO-CG"])
plt.plot(timelistpzo, ydislistpzo,  label='PZOBO', **styles["PZOBO"])


plt.xlabel('Running time (s)', fontsize=30)
plt.ylabel(r'$||y-y*||/||y*||$', fontsize=30, fontweight=900)
plt.xticks(fontsize=20)
plt.xticks(ticks)
plt.xlim(-0.05,3)

plt.yticks(fontsize=20)
plt.yscale('log')
plt.grid(visible=True, which='major', linestyle='-.', alpha=0.7)

plt.savefig('ydis.pdf', dpi=300, bbox_inches='tight')


legend_elements = [mlines.Line2D([], [], label=label, **style) for label, style in styles.items()]


legend_font = FontProperties(weight='heavy',size=20)
plt.figure(figsize=(3,0.5))
plt.legend(handles=legend_elements, ncol=6, fontsize=8, prop=legend_font, borderpad=1, loc='upper left',
           handlelength=2, handletextpad=2)



plt.axis('off')
plt.savefig('toylegend.pdf', dpi=300, bbox_inches='tight')