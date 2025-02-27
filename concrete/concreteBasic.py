import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import interpolate

class ColumnCircular:
    def __init__(self,D,rec,fc,beta_1,eco):
        self.D=D
        self.rec=rec
        self.fc=fc
        self.beta_1=beta_1
        self.eco=eco
        self.Ag=math.pi*self.D**2/4
        
    def Acero(self,fy,Es,numero_varillas,phi,phi_transversal,theta_i):
        self.fy=fy
        self.Es=Es
        self.ey=self.fy/self.Es
        self.numero_varillas=numero_varillas
        self.phi=phi/10 # Cambiamos de milimetros a centrimetros
        self.phi_transversal=phi_transversal/10 # Cambiamos de milimetros a centrimetros
        self.theta_i=theta_i
        self.omega=1.25 # Factor de Sobreresistencia 
        self.ey_omega=self.fy*self.omega/self.Es
        
        self.Calculos_Seccion()
        self.Calculos_ACI()
        
        self.c_array()
        self.DI=self.P_M_array(self.c_array)
        self.Mn_0,self.phi_Mn_0=self.Mn(0)
        self.Mn_lim,self.phi_Mn_lim=self.Mn(self.Pn_lim)
        
                      
    def Calculos_Seccion(self):
        
        self.d=self.D-self.rec-self.phi/2-self.phi_transversal
        self.r_varilla=(self.D-2*self.rec-2*self.phi_transversal-self.phi)/2
        self.delta_theta=360/self.numero_varillas
        self.d_array=self.Calculo_d_array(self.theta_i)
        self.As_array=self.Calculo_Capas_As()
        self.As=np.sum(self.As_array)
        self.rho=self.As/self.Ag
    
    def Calculo_d_array(self, theta_i):
        n_array=np.arange(0,self.numero_varillas,1)
        theta_array=n_array*(self.delta_theta+theta_i)
        y_array=self.r_varilla*np.cos(np.radians(theta_array))
        d_array=(self.r_varilla-y_array)+(self.phi/2+self.rec+self.phi_transversal)
        return d_array
    
    def Calculo_Capas_As(self):
        As_varilla=As(1,self.phi)
        As_array=np.full((1,self.numero_varillas),As_varilla)
        return As_array
    
    def Calculos_ACI(self):
        
        """
        Para llamar este metodo se debe tener instanciado __init__ / Armadura / Propiedades
        Es el metodo utilizado para realizar calculos generales de la seccion
            - Se calcula Po
            - Se calcula los valores asociados con el factor de reduccion de resistencia phi
        """

        Po=0.85*self.fc*(self.Ag-self.As)+self.fy*self.As #[kgf]
        self.Po=Po*10**-3 #[tf] 
        # 0.80 para estribos y 0.85 para espirales
        self.Pn_max=0.85*self.Po
        # 0.65 para estribos y 0.75 para espirales
        self.phi_Pn_max=0.75*self.Pn_max
        
        self.strain_phi_ACI=np.array([0,-self.ey,-self.eco-self.ey,-self.eco-2*self.ey])
        self.phi_ACI=np.array([0.65,0.65,0.9,0.9])
        self.cb=(self.eco)/(self.eco+self.ey)*self.d
        self.Calculos_Balanceados()
        self.Pn_lim=0.10*self.fc*self.Ag*10**-3
    
    def es_calculado(self,c,x):
        """
        Funcion para calcular la deformacion unitaria en el punto x para un valor del eje neutro de c

        Parameters
        ----------
        c : TYPE
            Distancia del Eje Neutro.
        x : TYPE
            Distancia de la fibra extrema en compresion al punto donde se busca calcular la deformacion unitaria.

        Returns
        -------
        es_x : TYPE
            Deformacion unitaria en la distancia x para el valor de C.

        """
        es_x=(-self.eco/c)*x+self.eco
        return es_x
    
    def fs(self,es):
        """
        Calculo del esfuerzo de la varilla en funcion de la deformacion unitaria para un modelo constitutivo bilineal

        Parameters
        ----------
        es : TYPE
            Deformacion unitaria.

        Returns
        -------
        fs : TYPE
            Esfuerzo de la varilla.

        """
        if es<=-self.ey or es>=self.ey:
            fs=self.fy*es/abs(es)
        else:
            fs=self.Es*es
        return fs
    
    def fs_omega(self,es):
        """
        Calculo del esfuerzo de la varilla en funcion de la deformacion unitaria para un modelo constitutivo bilineal

        Parameters
        ----------
        es : TYPE
            Deformacion unitaria.

        Returns
        -------
        fs : TYPE
            Esfuerzo de la varilla.

        """
        if es<=-self.ey_omega or es>=self.ey_omega:
            fs=self.fy*self.omega*es/abs(es)
        else:
            fs=self.Es*es
        return fs
    
    def fs_d(self,es):
        fs_d=np.vectorize(self.fs)
        return fs_d(es)
    
    def fs_d_omega(self,es):
        fs_d_omega=np.vectorize(self.fs_omega)
        return fs_d_omega(es)
    
    def es_d(self,c):
        """
        Funcion para calcular las deformaciones unitarias a las distancias d de las varillas (medidas desde la fibra extrema en compresion) para un valor del eje neutro igual a c

        Parameters
        ----------
        c : TYPE
            Distancia del eje neuto.

        Returns
        -------
        es_d : TYPE
            Deformacion unitaria para d en funcion de c.

        """
        es_d=self.es_calculado(c,self.d_array)
        return es_d
    
    def A_compresion(self,c):
        """
        Funcion para calcular el area de compresion y el centroide en funcion de la posicion de c

        Parameters
        ----------
        c : TYPE
            Distancia del eje neutro.

        Returns
        -------
        A_compresion : TYPE
            Distancia de a.

        """
        a=self.beta_1*c
        if a>=self.D:
            A_compresion=self.Ag
            A_centroide=self.D/2
        elif a<=self.D/2:
            theta=math.acos((self.D/2-a)/(self.D/2))
            A_compresion=self.D**2*((theta-np.sin(theta)*np.cos(theta))/(4))
            A_centroide=(self.D**3*((np.sin(theta))**3/(12)))/A_compresion
            A_centroide=self.D/2-A_centroide
        else:
            phi=math.acos((a-self.D/2)/(self.D/2))
            theta=math.pi-phi
            A_compresion=self.D**2*((theta-np.sin(theta)*np.cos(theta))/(4))
            A_centroide=(self.D**3*((np.sin(theta))**3/(12)))/A_compresion
            A_centroide=self.D/2-A_centroide
            
            
        return A_compresion,A_centroide
    
    def A_compresion_array(self,c):
        A_compresion_array=np.vectorize(self.A_compresion)
        return A_compresion_array(c)
    
    def phi_calculado(self,strain):
        e_interpol=interpolate.interp1d(self.strain_phi_ACI,self.phi_ACI,fill_value="extrapolate")
        phi=e_interpol(strain)
        return phi
    
    def P_M_c(self,c):
        # Calculo de las fuerzas y centroide del concreto
        Ac,y_c=self.A_compresion(c)
        Pc=0.85*Ac*self.fc/1000 # [tf]
        Mc=Pc*y_c/100 #[tf.m]
        # Calculo de Fuerzas en el Acero
        es_d=self.es_d(c)
        fs_d=self.fs_d(es_d)
        fs_d_omega=self.fs_d_omega(es_d)
        Fs_d=fs_d*self.As_array/1000 # [tf]
        Fs_d_omega=fs_d_omega*self.As_array/1000
        Ms_d=Fs_d*self.d_array/100 #[tf.m]
        Ms_d_omega=Fs_d_omega*self.d_array/100
        Ps=np.sum(Fs_d) #[tf]
        Ps_omega=np.sum(Fs_d_omega)
        Ms=np.sum(Ms_d) #[tf.m]
        Ms_omega=np.sum(Ms_d_omega) #[tf.m
        # Calculo de Pn y Mn
        Pn=Pc+Ps
        Pn_omega=Pc+Ps_omega
        Mn=Pn*self.D/200-(Mc+Ms)
        Mn_omega=Pn_omega*self.D/200-(Mc+Ms_omega)
        phi=self.phi_calculado(np.min(es_d))
        phi_Pn=min(phi*Pn,self.phi_Pn_max)
        
        
        
        return c,phi.item(),Pn,Mn,phi_Pn,phi*Mn,Pn_omega,Mn_omega
    
    def P_M_array(self,c):
        P_M_array=np.vectorize(self.P_M_c)
        return P_M_array(c)
    
    def c_array(self):
        c1=np.linspace(0.001,self.cb,num=21)
        c2=np.linspace(self.cb,self.D*1.5,9)
        c3=np.array([10*10**5])
        self.c_array=np.concatenate((c1,c2,c3),axis=0)
    
    def Calculos_Balanceados(self):
        _,_,self.Pn_bal,self.Mn_bal,_,_,_,_=self.P_M_c(self.cb)
        
    def Mn(self,Pn):
        interpolar=interpolate.interp1d(self.DI[2], self.DI[3])
        interpolar_phi=interpolate.interp1d(self.DI[4], self.DI[5])
        Mn=interpolar(Pn)
        phi_Mn=interpolar_phi(Pn)
        return Mn,phi_Mn

class ColumnRectangular:
    def __init__(self,B,H,rec,fc,beta_1,eco):
        self.B=B
        self.H=H
        self.rec=rec
        self.fc=fc
        self.beta_1=beta_1
        self.eco=eco
        self.Ag=self.B*self.H
        
    def Acero(self,fy,Es,numero_varillas_B,numero_varillas_H,phi_esquinas,phi_centro,phi_transversal):
        self.fy=fy
        self.Es=Es
        self.ey=self.fy/self.Es
        self.numero_varillas_B=numero_varillas_B
        self.numero_varillas_H=numero_varillas_H
        self.numero_varillas=self.numero_varillas_B*2+(self.numero_varillas_H-2)*2
        self.phi_esquinas=phi_esquinas/10 # Cambiamos de milimetros a centrimetros
        self.phi_centro=phi_centro/10 # Cambiamos de milimetros a centrimetros
        self.phi_transversal=phi_transversal/10 # Cambiamos de milimetros a centrimetros
        self.omega=1.25 # Factor de Sobreresistencia 
        self.ey_omega=self.fy*self.omega/self.Es
        
        self.Calculos_Seccion()
        self.Calculos_ACI()
        
        self.c_array()
        self.DI=self.P_M_array(self.c_array)
        self.Mn_0,self.phi_Mn_0=self.Mn(0)
        self.Mn_lim,self.phi_Mn_lim=self.Mn(self.Pn_lim)
        
                      
    def Calculos_Seccion(self):
        
        self.phi_longitudinal=max(self.phi_centro,self.phi_esquinas)
        self.d=self.H-self.rec-self.phi_longitudinal/2-self.phi_transversal
        self.d_array=np.linspace(self.rec+self.phi_transversal+self.phi_longitudinal/2,self.d,self.numero_varillas_H)
        self.As_array=self.Calculo_Capas_As()
        self.As=np.sum(self.As_array)
        self.rho=self.As/self.Ag
    
    def Calculo_Capas_As(self):
        As_capa_TOP_BOT=As(2,self.phi_esquinas)+As(self.numero_varillas_B-2,self.phi_centro)
        As_capa_CENTRO=As(2,self.phi_centro)
        As_array=np.zeros(self.d_array.shape[0])
        As_array[[0,-1]]=As_capa_TOP_BOT
        As_array[1:-1]=As_capa_CENTRO
        return As_array
    
    def Calculos_ACI(self):
        
        """
        Para llamar este metodo se debe tener instanciado __init__ / Armadura / Propiedades
        Es el metodo utilizado para realizar calculos generales de la seccion
            - Se calcula Po
            - Se calcula los valores asociados con el factor de reduccion de resistencia phi
        """

        Po=0.85*self.fc*(self.Ag-self.As)+self.fy*self.As #[kgf]
        self.Po=Po*10**-3 #[tf] 
        self.Pn_max=0.80*self.Po
        self.phi_Pn_max=0.65*self.Pn_max
        
        self.strain_phi_ACI=np.array([0,-self.ey,-self.eco-self.ey,-self.eco-2*self.ey])
        self.phi_ACI=np.array([0.65,0.65,0.9,0.9])
        self.cb=(self.eco)/(self.eco+self.ey)*self.d
        self.Calculos_Balanceados()
        self.Pn_lim=0.10*self.fc*self.Ag*10**-3
    
    def es_calculado(self,c,x):
        """
        Funcion para calcular la deformacion unitaria en el punto x para un valor del eje neutro de c

        Parameters
        ----------
        c : TYPE
            Distancia del Eje Neutro.
        x : TYPE
            Distancia de la fibra extrema en compresion al punto donde se busca calcular la deformacion unitaria.

        Returns
        -------
        es_x : TYPE
            Deformacion unitaria en la distancia x para el valor de C.

        """
        es_x=(-self.eco/c)*x+self.eco
        return es_x
    
    def fs(self,es):
        """
        Calculo del esfuerzo de la varilla en funcion de la deformacion unitaria para un modelo constitutivo bilineal

        Parameters
        ----------
        es : TYPE
            Deformacion unitaria.

        Returns
        -------
        fs : TYPE
            Esfuerzo de la varilla.

        """
        if es<=-self.ey or es>=self.ey:
            fs=self.fy*es/abs(es)
        else:
            fs=self.Es*es
        return fs
    
    def fs_omega(self,es):
        """
        Calculo del esfuerzo de la varilla en funcion de la deformacion unitaria para un modelo constitutivo bilineal

        Parameters
        ----------
        es : TYPE
            Deformacion unitaria.

        Returns
        -------
        fs : TYPE
            Esfuerzo de la varilla.

        """
        if es<=-self.ey_omega or es>=self.ey_omega:
            fs=self.fy*self.omega*es/abs(es)
        else:
            fs=self.Es*es
        return fs
    
    def fs_d(self,es):
        fs_d=np.vectorize(self.fs)
        return fs_d(es)
    
    def fs_d_omega(self,es):
        fs_d_omega=np.vectorize(self.fs_omega)
        return fs_d_omega(es)
    
    def es_d(self,c):
        """
        Funcion para calcular las deformaciones unitarias a las distancias d de las varillas (medidas desde la fibra extrema en compresion) para un valor del eje neutro igual a c

        Parameters
        ----------
        c : TYPE
            Distancia del eje neuto.

        Returns
        -------
        es_d : TYPE
            Deformacion unitaria para d en funcion de c.

        """
        es_d=self.es_calculado(c,self.d_array)
        return es_d
    
    def A_compresion(self,c):
        """
        Funcion para calcular el area de compresion y el centroide en funcion de la posicion de c

        Parameters
        ----------
        c : TYPE
            Distancia del eje neutro.

        Returns
        -------
        A_compresion : TYPE
            Distancia de a.

        """
        a=self.beta_1*c
        if a>=self.H:
            A_compresion=self.Ag
            A_centroide=self.H/2
        else:
            A_compresion=self.B*a
            A_centroide=a/2
            
        return A_compresion,A_centroide
    
    def A_compresion_array(self,c):
        A_compresion_array=np.vectorize(self.A_compresion)
        return A_compresion_array(c)
    
    def phi_calculado(self,strain):
        e_interpol=interpolate.interp1d(self.strain_phi_ACI,self.phi_ACI,fill_value="extrapolate")
        phi=e_interpol(strain)
        return phi
    
    def P_M_c(self,c):
        # Calculo de las fuerzas y centroide del concreto
        Ac,y_c=self.A_compresion(c)
        Pc=0.85*Ac*self.fc/1000 # [tf]
        Mc=Pc*y_c/100 #[tf.m]
        # Calculo de Fuerzas en el Acero
        es_d=self.es_d(c)
        fs_d=self.fs_d(es_d)
        fs_d_omega=self.fs_d_omega(es_d)
        Fs_d=fs_d*self.As_array/1000 # [tf]
        Fs_d_omega=fs_d_omega*self.As_array/1000
        Ms_d=Fs_d*self.d_array/100 #[tf.m]
        Ms_d_omega=Fs_d_omega*self.d_array/100
        Ps=np.sum(Fs_d) #[tf]
        Ps_omega=np.sum(Fs_d_omega)
        Ms=np.sum(Ms_d) #[tf.m]
        Ms_omega=np.sum(Ms_d_omega) #[tf.m
        # Calculo de Pn y Mn
        Pn=Pc+Ps
        Pn_omega=Pc+Ps_omega
        Mn=Pn*self.H/200-(Mc+Ms)
        Mn_omega=Pn_omega*self.H/200-(Mc+Ms_omega)
        phi=self.phi_calculado(np.min(es_d))
        phi_Pn=min(phi*Pn,self.phi_Pn_max)
        
        
        
        return c,phi.item(),Pn,Mn,phi_Pn,phi*Mn,Pn_omega,Mn_omega
    
    def P_M_array(self,c):
        P_M_array=np.vectorize(self.P_M_c)
        return P_M_array(c)
    
    def c_array(self):
        c1=np.linspace(0.001,self.cb,num=21)
        c2=np.linspace(self.cb,self.H*1.5,9)
        c3=np.array([10*10**5])
        self.c_array=np.concatenate((c1,c2,c3),axis=0)
    
    def Calculos_Balanceados(self):
        _,_,self.Pn_bal,self.Mn_bal,_,_,_,_=self.P_M_c(self.cb)
        
    def Mn(self,Pn):
        interpolar=interpolate.interp1d(self.DI[2], self.DI[3])
        interpolar_phi=interpolate.interp1d(self.DI[4], self.DI[5])
        Mn=interpolar(Pn)
        phi_Mn=interpolar_phi(Pn)
        return Mn,phi_Mn
    
    
class BeamSimple:
    def __init__(self,B,H,rec,phi_transversal,phi_longitudinal,fc,eco,fy,Es):
        self.B=B
        self.H=H
        self.rec=rec
        self.phi_transversal=phi_transversal/10
        self.phi_longitudinal=phi_longitudinal/10
        self.fc=fc
        self.fy=fy
        self.eco=eco
        self.Es=Es
        self.d=H-rec-self.phi_transversal-self.phi_longitudinal/2
        self.Ag=self.B*self.H
        self.Acero_Minimo()
        self.rho_max=0.025
        self.ey=self.fy/self.Es
        self.Calculo_Balanceado()
        
    def Acero_Minimo(self):
        rho_min_1=14/self.fy
        rho_min_2=0.80*math.sqrt(self.fc)/self.fy
        self.rho_min=max(rho_min_1,rho_min_2)
    
    def Calculo_Balanceado(self):
        self.cb=(self.eco)/(self.eco+self.ey)*self.d
        self.rho_bal=(self.eco)/(self.eco+self.ey)*(0.85**2*self.fc)/self.fy
        
    def Mn(self,rho):
        Mn=self.fy*(rho*self.B*self.d)*(self.d-(self.fy*(rho*self.B*self.d))/(1.70*self.fc*self.B))*10**-5
        return Mn
    
    def Mn_rho(self,rho):
        Mn_rho=np.vectorize(self.Mn)
        return Mn_rho(rho)
    
# Funcion para calcular el area de acero de un grupo de varillas
def As(n,phi):
    """
    

    Parameters
    ----------
    n : TYPE
        Cantidad de Varillas.
    phi : TYPE
        Diametro de las varillas.

    Returns
    -------
    As : TYPE
        Area de Acero en las unidades de la varilla.

    """
    As=n*math.pi*phi**2/4
    return As

def rho_table(rho,fraction):
    rho_sup=rho
    rho_inf=rho*fraction
    rho_inf[rho_inf<=rho_sup[0]]=rho_sup[0]
    rho_table=np.array([rho_sup,rho_inf])
    return rho_table

def Calculo_Ratio(Viga,fraction,Columna,P_sup,P_inf):
    """
    Calculo de ratio Columna Fuerte/Viga Debil

    Parameters
    ----------
    Viga : TYPE
        Clase de la Viga.
    fraction : TYPE
        Fraccion: As_negativo/As_positivo.
    Columna : TYPE
        Clase de la Columna.
    P_sup : TYPE
        Calga axial en la columna del nudo superior.
    P_inf : TYPE
        Calga axial en la columna del nudo inferior.

    Returns
    -------
    TYPE
        Cuantia del acero para momento negativo.
    ratio : TYPE
        Ratio Columna Fuerte/Viga Debil.
    fraction : TYPE
        Porcentaje del As Negativo/As Positivo.

    """
    
    rho=np.linspace(Viga.rho_min, Viga.rho_max,100)
    rho_viga=rho_table(rho, fraction)
    Mn_vigas=Viga.Mn_rho(rho_viga[0])+Viga.Mn_rho(rho_viga[1])
    
    # Calculamos el momento nominal de la columna que entra al nudo
    Mn_col_sup,_=Columna.Mn(P_sup)
    Mn_col_inf,_=Columna.Mn(P_inf)
    Mn_columna=Mn_col_sup+Mn_col_inf
    
    # Calculamos el ratio de Capacidad
    ratio=Mn_columna/Mn_vigas
    
    # Para corregir para el etabs se multiplica por d/H
    
    return rho_viga[0]*Viga.d/Viga.H, ratio, fraction