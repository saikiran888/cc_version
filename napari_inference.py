import pathlib
from magicgui import magicgui
import time, os
from magicgui.widgets import *


@magicgui(
          czi={'mode': 'r', 'label': 'Select CZI file'},
          output={'mode': 'd', 'label': 'Extract to'},
          sblk={'label':'Starting Block', 'min':-1},
          eblk={'label':'Ending Block', 'min':-1},
          sframe={'label':'Starting Frame', 'min':-1},
          eframe={'label':'Ending Frame', 'min':-1},
          call_button='Extract CZI to tiff')

def EXTRACT(   czi = pathlib.Path('/cellchorus/data/d/microscope/Effector_only/20220525_LC_NK_PKHGreen-01.czi'),
          output = pathlib.Path('/cellchorus/data/d/microscope/Effector_only/'),
          sblk = -1,
          eblk = -1,
          sframe = -1,
          eframe = -1,
          ):
       
    dataId=str(czi.name)[:-4]
    parent = str(czi.parent)+"/"
    output = str(output)+"/"
    
    sf=int(sframe)
    ef=int(eframe)
    sb=int(sblk)
    eb=int(eblk)
    
    print("""
            dataId = {}
            parent = {} 
            output = {}""".format(dataId,parent,output))
    os.system("python czi2tif.py --raw {} --output {} --dataset {} -sf {} -ef {} -sb {} -eb {} ".format(parent, output, dataId, sf, ef, sb, eb))
    
    
@magicgui(raw={'mode': 'd', 'label':'Data Dir'},
          result={'mode': 'd', 'label': 'Result Dir'},
          dataset={'mode': 'd', 'label': 'Dataset Dir'},
          sblk={'label':'Starting Block', 'min':0},
          eblk={'label': 'Ending Block', 'min':0},
          frame={'label': 'No. of Frames', 'min':0},
          version={"choices": ['40micron_v1_oct21', '50micron_v1_oct21', '50micron_v2_feb23','2TPS_april2024','beads_version','sandbox','kyvernaCAR','DT_may2024','SRK','SRK-ND2']},  call_button=True)
def DT(raw =  pathlib.Path.home(),
           result =  pathlib.Path.home(),
           dataset = pathlib.Path.home(),
           sblk = 1,
           eblk = 2,
           frame = 73,
           seg = False,
           overwrite = False,
           change_channels = False,
           cc_threshold = 0.7,
           version = '50micron_v1_oct21'
          ):
    a=str(raw)+"/"
    b=str(result)+"/"
    dataId=str(dataset.name)
    sb=int(sblk)
    eb=int(eblk)
    fr=int(frame)
    s="yes" if seg else "no"
    o="yes" if overwrite else "no"
    x="yes" if change_channels else "no"
    c= cc_threshold
    h = "/cellchorus/src/multiversion_testing/DEEP-TIMING/"+str(version)+"/"
    #h = "/home/sagar/DEEP-TIMING/"+str(version)+"/"
    print("""
            raw = {}
            reasult = {}
            dataid = {} 
            sb = {}
            eb = {}
            dt = {}
            s = {}
            o = {}
            x = {}
            c = {}
            h = {}""".format(a,b,dataId,sb,eb,fr,s,o,x,c,h))
    #for i in range(sb, eb+1):
     #   blk = "B"+str(i).zfill(3)
        #result.value = blk
    os.system("./run_pipeline.sh -h {} -a {} -b {} -d {} -s {} -e {} -n {} -m {} -o {} -x {} -c {}".format(h,a,b,dataId,sb,eb,fr,s,o,x,c))
        #print("Processing "+ blk + " ...")
        #time.sleep(1)

container = Container(widgets=[EXTRACT, DT])
container.show(run=True)
# widget.show()
