import os,sys,pdb,shutil
import numpy as np
import argparse

class ABNORMAL(object):
    def __init__(self, filepath, outdir, th=0.001):
        abnlist = []
        with open(filepath,'rb') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    continue
                src = os.path.splitext(line.split('|')[0])[0] + '.jpg'
                pr = np.float64( line.split('|')[-1] )
                if pr < th:
                    abnlist.append(src)
        try:
            os.makedirs(outdir)
        except Exception, e:
            pass
        for path in abnlist:
            shutil.move(path, outdir)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("infile", help="test.log from abndet.py test mode")
    ap.add_argument("-th",help="threshold of pr, [0,1]", default=0.001)
    ap.add_argument('outdir',help='output dir')
    args = ap.parse_args()
    ABNORMAL(args.infile, args.outdir, args.th)
