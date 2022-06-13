from unittest import result
import numpy as np
mW = 80.38
def EquationSolve(a,b,c,d):
    result=None
  if (a != 0) {
    
    T q = (3*a*c-b*b)/(9*a*a);
    T r = (9*a*b*c - 27*a*a*d - 2*b*b*b)/(54*a*a*a);
    T Delta = q*q*q + r*r;

    std::complex<T> s;
    std::complex<T> t;

    T rho=0;
    T theta=0;
    
    if( Delta<=0){
      rho = sqrt(-(q*q*q));

      theta = acos(r/rho);

      s = std::polar<T>(sqrt(-q),theta/3.0); 
      t = std::polar<T>(sqrt(-q),-theta/3.0); 
    }
    
    if(Delta>0){ 
      s = std::complex<T>(cbrt(r+sqrt(Delta)),0);
      t = std::complex<T>(cbrt(r-sqrt(Delta)),0);
    }
  
    std::complex<T> i(0,1.0); 
    
    
     x1 = s+t+std::complex<T>(-b/(3.0*a),0);
     x2 = (s+t)*std::complex<T>(-0.5,0)-std::complex<T>(b/(3.0*a),0)+(s-t)*i*std::complex<T>(sqrt(3)/2.0,0);
     x3 = (s+t)*std::complex<T>(-0.5,0)-std::complex<T>(b/(3.0*a),0)-(s-t)*i*std::complex<T>(sqrt(3)/2.0,0);

    if(fabs(x1.imag())<0.0001)result.push_back(x1.real());
    if(fabs(x2.imag())<0.0001)result.push_back(x2.real());
    if(fabs(x3.imag())<0.0001)result.push_back(x3.real());

    return result;
  }
  else :return result


  return result;
def compute_topmass(lep,met):
        EquationA = 1
        EquationB = -3*lep.py*mW/(lep.pt)
        EquationC = mW*mW*(2*lep.py*lep.py)/(lep.pt*lep.pt)+mW*mW-4*lep.px*lep.px*lep.px*met.px/(lep.pt*lep.pt)-4*lep.px*lep.px*lep.py*met.py/(lep.pt*lep.pt)
        EquationD = 4*lep.px*lep.px*mW*met.py/(lep.pt)-lep.py*mW*mW*mW/lep.pt
        deltaMin = 14000*14000
        zeroValue = -mW*mW/(4*lep.px)
        minPx=0
        minPy=0

        std::vector<long double> solutions = EquationSolve<long double>((long double)EquationA,(long double)EquationB,(long double)EquationC,(long double)EquationD)
        std::vector<long double> solutions2 = EquationSolve<long double>((long double)EquationA,-(long double)EquationB,(long double)EquationC,-(long double)EquationD)


        for( int i =0 i< (int)solutions.size()++i){
            if(solutions[i]<0 ) continue
            p_x = (solutions[i]*solutions[i]-mW*mW)/(4*lep.px)
            p_y = ( mW*mW*lep.py + 2*lep.px*lep.py*p_x -mW*lep.pt*solutions[i])/(2*lep.px*lep.px)
            Delta2 = (p_x-met.px)*(p_x-met.px)+(p_y-met.py)*(p_y-met.py)

            if(Delta2< deltaMin && Delta2 > 0){deltaMin = Delta2
            minPx=p_x
            minPy=p_y}
        }

        for( int i =0 i< (int)solutions2.size()++i){
            if(solutions2[i]<0 ) continue
            p_x = (solutions2[i]*solutions2[i]-mW*mW)/(4*lep.px)
            p_y = ( mW*mW*lep.py + 2*lep.px*lep.py*p_x +mW*lep.pt*solutions2[i])/(2*lep.px*lep.px)
            Delta2 = (p_x-met.px)*(p_x-met.px)+(p_y-met.py)*(p_y-met.py)
            if(Delta2< deltaMin && Delta2 > 0){deltaMin = Delta2
            minPx=p_x
            minPy=p_y
            }
        }

        pyZeroValue= ( mW*mW*lep.px + 2*lep.px*lep.py*zeroValue)
        delta2ZeroValue= (zeroValue-met.px)*(zeroValue-met.px) + (pyZeroValue-met.py)*(pyZeroValue-met.py)

        if(deltaMin==14000*14000) return TLorentzVector(0,0,0,0)

        if(delta2ZeroValue < deltaMin){
            deltaMin = delta2ZeroValue
            minPx=zeroValue
            minPy=pyZeroValue}

        mu_Minimum = mW**2/2 + minPx*lep.px + minPy*lep.py
        a_Minimum  = (mu_Minimum*lep.pz)/(lep.energy*lep.energy - lep.pz*lep.pz)
        pznu = a_Minimum

        Enu = np.sqrt(minPx*minPx+minPy*minPy + pznu*pznu)
        p4nu_rec.SetPxPyPzE(minPx, minPy, pznu , Enu)
def getnu4vec(lep,met):
    
    MisET2=met.pt**2
    mu = mW**2/2+met.px*lep.px+met.py*lep.py
    a = mu*lep.pz/(lep.energy**2-lep.pz**2)
    
    mu = mW**2/2 + met.px*lep.px + met.py*lep.py
    a  = (mu*lep.pz)/(lep.energy*lep.energy - lep.pz*lep.pz)
    a2 = a**2
    b  = (lep.energy**2*MisET2 - mu**2)/(lep.energy,2 - lep.pz**2)


    p4lep_rec=lep
    p4nu_rec = met
    p4nu_rec.px = met.px
    p4nu_rec.py = met.py
    p4nu_rec.pz = np.where(a2-b>0,max(abs(a-np.sqrt(a2-b)),abs(a+np.sqrt(a2-b))),compute_topmass(lep,met))
    p4nu_rec.energy = np.where(a2-b>0,np.sqrt(max(abs(a-np.sqrt(a2-b)),abs(a+np.sqrt(a2-b)))**2+met.pt**2),compute_topmass(lep,met))
    
    
    return p4nu_rec