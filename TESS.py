import numpy as np
import matplotlib.pyplot as plt
import requests
import astropy.io.ascii as ascii
import math
from astropy import units as u
from astropy.coordinates import Angle
from astroquery.mast import Catalogs
from astropy.table import Table

TOI=requests.get("https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toid&output=csv")
table=ascii.read(TOI.text,delimiter=",",data_start=1)

TICid=np.array(table["col1"]) #TIC id
TOIid=np.array(table["col2"]) #TOI id
Period=np.array(table["col25"]) #days
dPeriod=np.array(table["col26"]) #days
R=np.array(table["col33"]) #R_earth
dR=np.array(table["col34"]) #R_earth
RA=np.array(table["col17"]) #RA hh:mm:ss.s
Dec=np.array(table["col18"]) #Dec +-Deg:mm:ss.s
Insol=np.array(table["col35"]) #EarthFlux
EqTemp=np.array(table["col36"]) #K
Depth=np.array(table["col31"]) #ppm
dDepth=np.array(table["col32"]) #ppm
EffTemp=np.array(table["col40"]) #K
dEffTemp=np.array(table["col41"]) #K
StellarR=np.array(table["col44"]) #R_sun
dStellarR=np.array(table["col45"]) #R_sun
Comments=np.array(table["col49"]) #Comments

indexA=np.where(Period==0)
TICid=np.delete(TICid,indexA)
TOIid=np.delete(TOIid,indexA)
Period=np.delete(Period,indexA)
dPeriod=np.delete(dPeriod,indexA)
R=np.delete(R,indexA)
dR=np.delete(dR,indexA)
RA=np.delete(RA,indexA)
Dec=np.delete(Dec,indexA)
Insol=np.delete(Insol,indexA)
EqTemp=np.delete(EqTemp,indexA)
Depth=np.delete(Depth,indexA)
dDepth=np.delete(dDepth,indexA)
EffTemp=np.delete(EffTemp,indexA)
dEffTemp=np.delete(dEffTemp,indexA)
StellarR=np.delete(StellarR,indexA)
dStellarR=np.delete(dStellarR,indexA)
Comments=np.delete(Comments,indexA)

indexB=np.where(R==0)
TICid=np.delete(TICid,indexB)
TOIid=np.delete(TOIid,indexB)
Period=np.delete(Period,indexB)
dPeriod=np.delete(dPeriod,indexB)
R=np.delete(R,indexB)
dR=np.delete(dR,indexB)
RA=np.delete(RA,indexB)
Dec=np.delete(Dec,indexB)
Insol=np.delete(Insol,indexB)
EqTemp=np.delete(EqTemp,indexB)
Depth=np.delete(Depth,indexB)
dDepth=np.delete(dDepth,indexB)
EffTemp=np.delete(EffTemp,indexB)
dEffTemp=np.delete(dEffTemp,indexB)
StellarR=np.delete(StellarR,indexB)
dStellarR=np.delete(dStellarR,indexB)
Comments=np.delete(Comments,indexB)


sorted=TICid.argsort()
TICid=TICid[sorted]
TOIid=TOIid[sorted]
Period=Period[sorted]
dPeriod=dPeriod[sorted]
R=R[sorted]
dR=dR[sorted]
RA=RA[sorted]
Dec=Dec[sorted]
Insol=Insol[sorted]
EqTemp=EqTemp[sorted]
Depth=Depth[sorted]
dDepth=dDepth[sorted]
EffTemp=EffTemp[sorted]
dEffTemp=dEffTemp[sorted]
StellarR=StellarR[sorted]
dStellarR=dStellarR[sorted]
Comments=Comments[sorted]

uniqueTICid,uniqueindex=np.unique(TICid,return_index=True)


def Omega(R,Period):
    return R/Period**(1/3)

Omega=Omega(R,Period) #Omega parameter
dOmega=np.sqrt((dR/Period**(1/3))**2+(R/3/Period**(4/3)*dPeriod)**2)

catalogData = Catalogs.query_criteria(catalog="Tic",ID=TICid)

catalogVmag=np.array(catalogData["Vmag"])
catalogdVmag=np.array(catalogData["e_Vmag"])

catalogJmag=np.array(catalogData["Jmag"])
catalogdJmag=np.array(catalogData["e_Jmag"])

catalogHmag=np.array(catalogData["Hmag"])
catalogdHmag=np.array(catalogData["e_Hmag"])

Vmag=np.zeros(len(TICid))
dVmag=np.zeros(len(TICid))
Jmag=np.zeros(len(TICid))
dJmag=np.zeros(len(TICid))
Hmag=np.zeros(len(TICid))
dHmag=np.zeros(len(TICid))

Vmag[uniqueindex]=catalogVmag
dVmag[uniqueindex]=catalogdVmag
Jmag[uniqueindex]=catalogJmag
dJmag[uniqueindex]=catalogdJmag
Hmag[uniqueindex]=catalogHmag
dHmag[uniqueindex]=catalogdHmag

for i in range(len(TICid)):
    if Vmag[i]==0: Vmag[i]=Vmag[i-1]
    if dVmag[i]==0: dVmag[i]=dVmag[i-1]
    if Jmag[i]==0: Jmag[i]=Jmag[i-1]
    if dJmag[i]==0: dJmag[i]=dJmag[i-1]
    if Hmag[i]==0: Hmag[i]=Hmag[i-1]
    if dHmag[i]==0: dHmag[i]=dHmag[i-1]


#######################################################

class AnnoteFinder(object):


    def __init__(self, xdata, ydata, annotes, ax=None, xtol=None, ytol=None):
        self.data = list(zip(xdata, ydata, annotes))
        if xtol is None:
            xtol = ((max(xdata) - min(xdata))/float(len(xdata)))
        if ytol is None:
            ytol = ((max(ydata) - min(ydata))/float(len(ydata)))
        self.xtol = xtol
        self.ytol = ytol
        if ax is None:
            self.ax = plt.gca()
        else:
            self.ax = ax
        self.drawnAnnotations = {}
        self.links = []

    def distance(self, x1, x2, y1, y2):

        return(math.sqrt((x1 - x2)**2 + (y1 - y2)**2))

    def __call__(self, event):

        if event.inaxes:

            clickX = event.xdata
            clickY = event.ydata
            if (self.ax is None) or (self.ax is event.inaxes):
                annotes = []
                for x, y, a in self.data:
                    if ((clickX-self.xtol < x < clickX+self.xtol) and
                            (clickY-self.ytol < y < clickY+self.ytol)):
                        annotes.append(
                            (self.distance(x, clickX, y, clickY), x, y, a))
                if annotes:
                    annotes.sort()
                    distance, x, y, annote = annotes[0]
                    self.drawAnnote(event.inaxes, x, y, annote)
                    for l in self.links:
                        l.drawSpecificAnnote(annote)

    def drawAnnote(self, ax, x, y, annote):

        if (x, y) in self.drawnAnnotations:
            markers = self.drawnAnnotations[(x, y)]
            for m in markers:
                m.set_visible(not m.get_visible())
            self.ax.figure.canvas.draw_idle()
        else:
            t = ax.text(x, y, " TOI %s" % (annote),)
            m = ax.scatter([x], [y], marker='s', c='c', zorder=100)
            t.set_bbox(dict(facecolor='white'))
            index,=np.where(TOIid==annote)
            index=index[0]
            print("--------------------")
            print("Planet parameters (TOI %s)" % (annote))
            print(" ")
            print("Radius : {0} +- {1}".format(R[index],dR[index]), " R_earth")
            print("Period : {0} +- {1}".format(Period[index],dPeriod[index]), " days")
            print("Planet insolation : {0}".format(Insol[index]), " Earth flux")
            print("Teq : {0}".format(EqTemp[index]), " K")
            print("Transit depth : {0}".format(Depth[index]), " ppm")
            print(" ")
            print("Stellar parameters (TIC {0})".format(TICid[index]))
            print(" ")
            print("RA : {0}".format(RA[index]))
            print("Dec : {0}".format(Dec[index]))
            print("V : {0} +- {1}".format(Vmag[index],dVmag[index]), " mag")
            print("J : {0} +- {1}".format(Jmag[index],dJmag[index]), " mag")
            print("H : {0} +- {1}".format(Hmag[index],dHmag[index]), " mag")
            print("Radius : {0} +- {1}".format(StellarR[index],dStellarR[index]), " R_sun")            
            print("Teff : {0} +- {1}".format(EffTemp[index],dEffTemp[index]), " K")
            print(" ")
            print("Comments : {0}".format(Comments[index]))
            print("--------------------")
            self.drawnAnnotations[(x, y)] = (t, m)
            self.ax.figure.canvas.draw_idle()

    def drawSpecificAnnote(self, annote):
        annotesToDraw = [(x, y, a) for x, y, a in self.data if a == annote]
        for x, y, a in annotesToDraw:
            self.drawAnnote(self.ax, x, y, a)
			
####################################################		

#Update section (takes time)

# bool=np.zeros(len(TICid))

# for i in range(len(TICid)):
    # print(i)
    # Info=requests.get("https://exofop.ipac.caltech.edu/tess/download_target.php?id={0}".format(TICid[i]))
    # Text=Info.text
    # if "Planet Name(s)              N/A" in Text:
        # bool[i]=0
    # else:
        # bool[i]=1

# print(bool)

#New TESS confirmed planet

# TESS=requests.get("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=exoplanets&select=pl_name,pl_facility&where=pl_facility%20like%20%27Transiting%20Exoplanet%20Survey%20Satellite%20(TESS)%27")
# table=ascii.read(TESS.text,delimiter=",",data_start=1)

# PlanetName=np.array(table["pl_name"]) #TESS confirmed planet name
# TOIconfirmed=np.zeros(len(PlanetName))
# for i in range(len(PlanetName)):
    # TESS=requests.get("https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/nph-nstedAPI?table=aliastable&objname={0}".format(PlanetName[i]))
    # table=ascii.read(TESS.text,delimiter=",",data_start=1)
    # AliasPlanetName=np.array(table["aliasdis"])
    # for alias in AliasPlanetName:
        # if str(alias).startswith("TOI") and len(str(alias).split())==2:
            # TOIlist=str(alias).split()
            # TOIconfirmed[i]=TOIlist[1]

# print(TOIconfirmed)


####################################################

#Confirmed planet system 25/01/2019

ConfirmedArray=np.array([ 1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,
  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  1,  1,  1,  0,  1,  0,
  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  0,  0,  1,
  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  1,  0,
  0,  1,  0,  0,  0,  0,  0,  0,  0,  1,  1,  0,  1,  0,  0,  0,  1,  1,
  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,
  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,
  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,
  0,  0,  0,  1,  1,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
  1,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  1,  1,  0,  0,  0,  0,  1,  0,  0,  0,  0,  0,  0,  0,
  0,  0,  0,  0,  0,  0,  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  0,
  0,  0,  0,  1,  0,  0,  0,  0,  0,  1,  0,  0,  0,  1,  0,  0,  0,  0,
  0,  0,  0,  1,  0,  0,  0,  0,  1,  0,  0,  1,  0,  0,  0,  0,  0,  0,
  0,  0,  0,])
  
#Confirmed TESS planets 30/01/2019

TESSconfirmed=np.array([144.01, 136.01])

####################################################

TESSIndex=np.zeros(len(TESSconfirmed),dtype=int)
for i in range(len(TESSconfirmed)):
    TESSIndex[i],=np.where(TOIid==TESSconfirmed[i])
    
KnownIndex,=np.where(ConfirmedArray==1)
UnknownIndex,=np.where(ConfirmedArray==0)
UnknownIndex=np.append(UnknownIndex,TESSIndex)


DecAngle=Angle(Dec,unit=u.deg)   
DecIndex,=np.where(DecAngle.deg>=-20)    
DecIndex=np.intersect1d(DecIndex,UnknownIndex) 


while True:
    Search=input("Search? y/n : ")
    if Search=="y":
        print(" ")
        TOIidinput=input("TOI id : ")
        TOIidinput=float(TOIidinput)
        if TOIidinput in TOIid:
            Search=1
            
        else:
            print("No TOI found or unknown period/radius.")
            continue
    if Search=="n":
        Search=0
        break
            
    if Search==1:

        index,=np.where(TOIid==TOIidinput)
        index=index[0]
	
        Vrange=np.linspace(4,10.7,2)


        Jrange=np.linspace(4,11.7,2)

        
        print("--------------------")
        print("Planet parameters (TOI {0})".format(TOIid[index]))
        print(" ")
        print("Radius : {0} +- {1}".format(R[index],dR[index]), " R_earth")
        print("Period : {0} +- {1}".format(Period[index],dPeriod[index]), " days")
        print("Planet insolation : {0}".format(Insol[index]), " Earth flux")
        print("Teq : {0}".format(EqTemp[index]), " K")
        print("Transit depth : {0}".format(Depth[index]), " ppm")
        print(" ")
        print("Stellar parameters (TIC {0})".format(TICid[index]))
        print(" ")
        print("RA : {0}".format(RA[index]))
        print("Dec : {0}".format(Dec[index]))
        print("V : {0} +- {1}".format(Vmag[index],dVmag[index]), " mag")
        print("J : {0} +- {1}".format(Jmag[index],dJmag[index]), " mag")
        print("H : {0} +- {1}".format(Hmag[index],dHmag[index]), " mag")
        print("Radius : {0} +- {1}".format(StellarR[index],dStellarR[index]), " R_sun")            
        print("Teff : {0} +- {1}".format(EffTemp[index],dEffTemp[index]), " K")
        print(" ")
        print("Comments : {0}".format(Comments[index]))
        print("--------------------")
    
        fig, ax = plt.subplots(2,1)
    
        ax[0].plot(Vmag,Omega,color="k",marker="s",linestyle="none",markersize=3,alpha=.5)
        ax[0].errorbar(Vmag[index],Omega[index],xerr=dVmag[index],yerr=dOmega[index],color="r",marker="s",markersize=5,linestyle="none",label="TOI {0}".format(TOIidinput),lw=2,capsize=3)
        ax[0].axvline(x=10.7,ymin=1.243/int(Omega[index]+dOmega[index]+2),color="k",linestyle="--")
        ax[0].plot(Vrange,0.09*Vrange+0.28,color="k",linestyle="--")

        ax[0].set_xlabel("V magnitude")
        ax[0].set_ylabel(r"$\Omega = r_p/P^{1/3}$")
        ax[0].set_xlim(4,16)
        ax[0].set_ylim(0,int(Omega[index]+dOmega[index]+2))

        ax[1].plot(Jmag,Omega,color="k",marker="s",linestyle="none",markersize=3,alpha=.5)
        ax[1].errorbar(Jmag[index],Omega[index],xerr=dJmag[index],yerr=dOmega[index],color="r",marker="s",markersize=5,linestyle="none",lw=2,capsize=3)
        ax[1].axvline(x=11.7,ymin=1.288/int(Omega[index]+dOmega[index]+2),color="k",linestyle="--")
        ax[1].plot(Jrange,0.14*Jrange-0.35,color="k",linestyle="--")

        ax[1].set_xlabel("J magnitude")
        ax[1].set_ylabel(r"$\Omega = r_p/P^{1/3}$")
        ax[1].set_xlim(4,16)
        ax[1].set_ylim(0,int(Omega[index]+dOmega[index]+2))
        ax[0].legend(loc="best",numpoints=1)
        fig.tight_layout()
        plt.show()
    else:
        print("Error")	
        continue

##############################
    
if Search==0:
    
    Vrange=np.linspace(4,10.7,2)
    Jrange=np.linspace(4,11.7,2)

    fig, ax = plt.subplots(2,1,figsize=(15,10))

    ax[0].plot(Vmag[UnknownIndex],Omega[UnknownIndex],color="k",marker="s",linestyle="none",markersize=3)
    ax[0].plot(Vmag[DecIndex],Omega[DecIndex],color="r",marker="s",linestyle="none",markersize=3)
    ax[0].plot(Vmag[TESSIndex],Omega[TESSIndex],color="y",marker="s",linestyle="none",markersize=3)
    ax[0].axvline(x=10.7,ymin=0.1243,color="k",linestyle="--")
    ax[0].plot(Vrange,0.09*Vrange+0.28,color="k",linestyle="--")
    af1 =  AnnoteFinder(Vmag[UnknownIndex],Omega[UnknownIndex], TOIid[UnknownIndex], ax=ax[0])

    ax[0].set_xlabel("V magnitude")
    ax[0].set_ylabel(r"$\Omega = r_p/P^{1/3}$")
    ax[0].set_xlim(4,16)
    ax[0].set_ylim(0,10)


    ax[1].plot(Jmag[UnknownIndex],Omega[UnknownIndex],color="k",marker="s",linestyle="none",markersize=3)
    ax[1].plot(Jmag[DecIndex],Omega[DecIndex],color="r",marker="s",linestyle="none",markersize=3)
    ax[1].plot(Jmag[TESSIndex],Omega[TESSIndex],color="y",marker="s",linestyle="none",markersize=3)
    ax[1].axvline(x=11.7,ymin=0.1288,color="k",linestyle="--")
    ax[1].plot(Jrange,0.14*Jrange-0.35,color="k",linestyle="--")
    af2 =  AnnoteFinder(Jmag[UnknownIndex],Omega[UnknownIndex], TOIid[UnknownIndex], ax=ax[1])

    ax[1].set_xlabel("J magnitude")
    ax[1].set_ylabel(r"$\Omega = r_p/P^{1/3}$")
    ax[1].set_xlim(4,16)
    ax[1].set_ylim(0,10)

    fig.canvas.mpl_connect('button_press_event', af1)
    fig.canvas.mpl_connect('button_press_event', af2)
    
    fig.tight_layout()
    plt.show()

    
 
    
data=Table({'TIC ID':TICid[DecIndex],'TOI':TOIid[DecIndex],'RA': RA[DecIndex],'Dec':Dec[DecIndex],'Period':Period[DecIndex],'Period error':dPeriod[DecIndex],'Transit Depth':Depth[DecIndex],'Transit Depth error':dDepth[DecIndex],'Planet Radius':R[DecIndex],'Planet Radius error':dR[DecIndex]\
,'Planet Insolation':Insol[DecIndex],'Planet Eq Temp':EqTemp[DecIndex],'Stellar Teff':EffTemp[DecIndex],'Stellar Teff error':dEffTemp[DecIndex],'Stellar Radius':StellarR[DecIndex],'Stellar Radius error':dStellarR[DecIndex],'V':Vmag[DecIndex],'V error':dVmag[DecIndex]\
,'J':Jmag[DecIndex],'J error':dJmag[DecIndex],'H':Hmag[DecIndex],'H error':dHmag[DecIndex],'Comments':Comments[DecIndex]},names=['TIC ID','TOI','RA','Dec','Period','Period error','Transit Depth','Transit Depth error','Planet Radius','Planet Radius error','Planet Insolation'\
,'Planet Eq Temp','Stellar Teff','Stellar Teff error','Stellar Radius','Stellar Radius error','V','V error','J','J error','H','H error','Comments'])
ascii.write(data, 'TOI.csv',delimiter=",")
print(data)
print("Fichier .csv produit")