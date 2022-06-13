import os
def read_xs(files):
    import json
    def findfile(name, path):
        for dirpath, dirname, filename in os.walk(path):
            if name in filename:
                return os.path.join(dirpath, name)
    filepath = findfile(files, str(os.getcwd()))
    
    #f = open(os.path.abspath(filepath))
    f = open('/afs/cern.ch/work/m/milee/Hpluscharm/Hpluscharm_processor/metadata/xsection.json')
    data = json.load(f)
    xs_dict={}
    for obj in data:
        xs_dict[obj['process_name']]=float(obj['cross_section'])
    return xs_dict
def scale_xs(hist,lumi,events,unscale=False,xsfile="xsection.json"):

    xs_dict = read_xs(xsfile)
    scales={}

    for key in events:
        if type(key) != str or key=="Data" : continue
        if unscale: 
            scales[key]=events[key]/xs_dict[key]*lumi
        else :scales[key]=xs_dict[key]*lumi/events[key]

    hist.scale(scales, axis="dataset")
    return hist
def collate(accumulator, mergemap):
    out = {}
    for group, names in mergemap.items():
        out[group] = processor.accumulate([v for k, v in accumulator.items() if k in names])
    return out
