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
          version={"choices": ['40micron_v1_oct21', '50micron_v1_oct21', '50micron_v2_feb23','2TPS_april2024','beads_version','sandbox','kyvernaCAR','DT_may2024','SRK','SRK-ND2']},
          # Processing options
          seg={'label': 'Segmentation', 'widget_type': 'CheckBox'},
          overwrite={'label': 'Overwrite', 'widget_type': 'CheckBox'},
          change_channels={'label': 'Change Channels', 'widget_type': 'CheckBox'},
          cc_threshold={'label': 'CC Threshold', 'min': 0.0, 'max': 1.0},
          # Pipeline step controls
          run_initiate={'label': 'Run Initiate Results', 'widget_type': 'CheckBox'},
          run_part1={'label': 'Run Block Part 1 (Cropping)', 'widget_type': 'CheckBox'},
          run_part2={'label': 'Run Block Part 2 (Cell Detection)', 'widget_type': 'CheckBox'},
          run_part3={'label': 'Run Block Part 3 (Tracking)', 'widget_type': 'CheckBox'},
          run_result={'label': 'Run Process Result', 'widget_type': 'CheckBox'},
          run_et_ratio={'label': 'Run ET-ratio', 'widget_type': 'CheckBox'},
          run_secondary={'label': 'Run Secondary Analysis', 'widget_type': 'CheckBox'},
          call_button=True)
def DT(raw =  pathlib.Path.home(),
           result =  pathlib.Path.home(),
           dataset = pathlib.Path.home(),
           sblk = 1,
           eblk = 196,
           frame = 73,
           seg = False,
           overwrite = True,
           change_channels = False,
           cc_threshold = 0.7,
           version = 'SRK',
           run_initiate = True,
           run_part1 = False,
           run_part2 = False,
           run_part3 = False,
           run_result = False,
           run_et_ratio = False,
           run_secondary = True
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
    
    # Convert boolean pipeline controls to strings
    i="yes" if run_initiate else "no"
    p1="yes" if run_part1 else "no"
    p2="yes" if run_part2 else "no"
    p3="yes" if run_part3 else "no"
    pr="yes" if run_result else "no"
    et="yes" if run_et_ratio else "no"
    sec="yes" if run_secondary else "no"
    
    print("""
            raw = {}
            result = {}
            dataid = {} 
            sb = {}
            eb = {}
            frame = {}
            seg = {}
            overwrite = {}
            change_channels = {}
            cc_threshold = {}
            home = {}
            Pipeline Steps:
            - Initiate Results: {}
            - Block Part 1: {}
            - Block Part 2: {}
            - Block Part 3: {}
            - Process Result: {}
            - ET-ratio: {}
            - Secondary Analysis: {}""".format(a,b,dataId,sb,eb,fr,s,o,x,c,h,i,p1,p2,p3,pr,et,sec))
    
    #for i in range(sb, eb+1):
     #   blk = "B"+str(i).zfill(3)
        #result.value = blk
    
    # Build the command string for debugging
    cmd = "./run_pipeline.sh -h {} -a {} -b {} -d {} -s {} -e {} -n {} -m {} -o {} -x {} -c {} -i {} -j {} -k {} -l {} -r {} -q {} -v {} -z dummy".format(
        h,a,b,dataId,sb,eb,fr,s,o,x,c,i,p1,p2,p3,pr,et,sec)
    
    print("Executing command:")
    print(cmd)
    print(f"DEBUG: run_secondary={run_secondary}, sec={sec}")
    print(f"DEBUG: Full command: {cmd}")
    os.system(cmd)
        #print("Processing "+ blk + " ...")
        #time.sleep(1)

container = Container(widgets=[EXTRACT, DT])
container.show(run=True)
# widget.show()
