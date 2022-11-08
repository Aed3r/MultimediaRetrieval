import numpy as np
from tqdm import tqdm
import database as db
import distance_functions as df
import ann_search as ann

dbmngr = db.DatabaseManager()
database_length = 380

def getTruthTable(type="simple"):
    meshes = dbmngr.get_all_with_extracted_features()

    if not meshes.alive:
        raise Exception("No meshes with extracted features found in the db. Run 'python main.py extract' first.")

    # mesh = dbmngr.get_by_path("data\\normalized\\Airplane\\61.off")
    for mesh in tqdm(meshes, desc='Computing TruthTable', ncols=130, total=dbmngr.get_mesh_count_with_features()):
        Qlabel = mesh['class']

        if type == "simple":
            res = df.find_best_matches(mesh, k=5)
        elif type == "ann":
            nn = ann.get_knn(mesh, k=5)

            res = []
            for i, x in enumerate(nn[0]):
                mesh = dbmngr.get_all_with_extracted_features()[x]
                mesh["distance"] = nn[1][i]
                res.append(mesh)
        
        TP = FP = 0
        # dbQsum: total number of positive shapes in db
        dbQsum = dbmngr.get_mesh_count_by_category(Qlabel)

        Rlabel = []
        for r in res:
            Rlabel.append(r['class'])
            if Qlabel == r['class']:
                TP += 1
            else:
                FP += 1
        FN = dbQsum - TP

        # dbRsum: total number of negative shapes in db
        Rlabeltype = set(Rlabel)
        dbRsum = 0
        for rlt in Rlabeltype:
            if Qlabel != rlt:
                dbRsum += dbmngr.get_mesh_count_by_category(r['class'])
        TN = dbRsum - FP
        TruthTable = [TP, FP, FN, TN]
        mesh["TruthTable"] = TruthTable
        dbmngr.update_one(mesh)
    return 0

def acc(DBlabel):
    # ACC = (TP + TN) / (TP + FN + FP + TN)
    ACC = [[0 for i in range(2)] for i in range(len(DBlabel))]
    for i in range(len(ACC)):
        ACC[i][0] = DBlabel[i]

    # get ACC by class
    meshes = dbmngr.get_all_with_extracted_features()
    for mesh in meshes:
        # TruthTable = [TP, FP, FN, TN]
        accm = (mesh['TruthTable'][0] + mesh['TruthTable'][3]) / \
              (mesh['TruthTable'][0]+mesh['TruthTable'][1]+mesh['TruthTable'][2]+mesh['TruthTable'][3])

        for i in range(len(ACC)):
             if mesh['class'] == ACC[i][0]:
                ACC[i][1] += accm

    # get ACC average by class
    ACCavg = [[0 for i in range(2)] for i in range(len(ACC))]
    for i in range(len(ACC)):
        ACCavg[i][0] = ACC[i][0]
        ACCavg[i][1] = ACC[i][1] / dbmngr.get_mesh_count_by_category(ACC[i][0])

    # get ACC for all class
    ACCsum = 0
    for i in range(len(ACC)):
        ACCsum += ACC[i][1]

    # get ACC avarage for all class
    ACCdbavg = ACCsum / database_length
    # print("ACC:", ACC)
    # print("ACCsum", ACCsum)
    print("average ACC of each class:", ACCavg)
    print("average ACC of the database:", ACCdbavg)

    return 0

def ppv(DBlabel):
    # PPV = TP / (TP + FP)
    PPV = [[0 for i in range(2)] for i in range(len(DBlabel))]
    for i in range(len(PPV)):
        PPV[i][0] = DBlabel[i]

    # get PPV by class
    meshes = dbmngr.get_all_with_extracted_features()
    for mesh in meshes:
        # TruthTable = [TP, FP, FN, TN]

        if mesh['TruthTable'][0] + mesh['TruthTable'][1] == 0:
            ppvm = 0
        else:
            ppvm = mesh['TruthTable'][0] / (mesh['TruthTable'][0] + mesh['TruthTable'][1])

        for i in range(len(PPV)):
             if mesh['class'] == PPV[i][0]:
                PPV[i][1] += ppvm

    # get PPV average by class
    PPVavg = [[0 for i in range(2)] for i in range(len(PPV))]
    for i in range(len(PPV)):
        PPVavg[i][0] = PPV[i][0]
        PPVavg[i][1] = PPV[i][1] / dbmngr.get_mesh_count_by_category(PPV[i][0])

    # get PPV for all class
    PPVsum = 0
    for i in range(len(PPV)):
        PPVsum += PPV[i][1]

    # get PPV avarage for all class
    PPVdbavg = PPVsum / database_length
    # print("PPV:", PPV)
    # print("PPVsum", PPVsum)
    print("average PPV of each class:", PPVavg)
    print("average PPV of the database:", PPVdbavg)
    return 0

def recall(DBlabel):
    # TPR = TP / (TP+FN)
    RECALL = [[0 for i in range(2)] for i in range(len(DBlabel))]
    for i in range(len(RECALL)):
        RECALL[i][0] = DBlabel[i]

    # get RECALL by class
    meshes = dbmngr.get_all_with_extracted_features()
    for mesh in meshes:
        # TruthTable = [TP, FP, FN, TN]
        recallm = mesh['TruthTable'][0] / (mesh['TruthTable'][0] + mesh['TruthTable'][2])

        for i in range(len(RECALL)):
             if mesh['class'] == RECALL[i][0]:
                RECALL[i][1] += recallm

    # get RECALL average by class
    RECALLavg = [[0 for i in range(2)] for i in range(len(RECALL))]
    for i in range(len(RECALL)):
        RECALLavg[i][0] = RECALL[i][0]
        RECALLavg[i][1] = RECALL[i][1] / dbmngr.get_mesh_count_by_category(RECALL[i][0])

    # get RECALL for all class
    RECALLsum = 0
    for i in range(len(RECALL)):
        RECALLsum += RECALL[i][1]

    # get RECALL avarage for all class
    RECALLdbavg = RECALLsum / database_length
    # print("RECALL:", RECALL)
    # print("RECALLsum", RECALLsum)
    print("average RECALL of each class:", RECALLavg)
    print("average RECALL of the database:", RECALLdbavg)
    return 0

def specificity(DBlabel):
    # Specificity =  TN / (FP + TN)
    SPEC = [[0 for i in range(2)] for i in range(len(DBlabel))]
    for i in range(len(SPEC)):
        SPEC[i][0] = DBlabel[i]

    # get SPEC by class
    meshes = dbmngr.get_all_with_extracted_features()
    for mesh in meshes:
        # TruthTable = [TP, FP, FN, TN]
        try:
            specm = mesh['TruthTable'][3] / (mesh['TruthTable'][1] + mesh['TruthTable'][3])

            for i in range(len(SPEC)):
                 if mesh['class'] == SPEC[i][0]:
                    SPEC[i][1] += specm
        except:
            continue
    # get SPEC average by class
    SPECavg = [[0 for i in range(2)] for i in range(len(SPEC))]
    for i in range(len(SPEC)):
        SPECavg[i][0] = SPEC[i][0]
        SPECavg[i][1] = SPEC[i][1] / dbmngr.get_mesh_count_by_category(SPEC[i][0])

    # get SPEC for all class
    SPECsum = 0
    for i in range(len(SPEC)):
        SPECsum += SPEC[i][1]

    # get SPEC avarage for all class
    SPECdbavg = SPECsum / database_length
    # print("SPEC:", SPEC)
    # print("SPECsum", SPECsum)
    print("average Specificity of each class:", SPECavg)
    print("average Specificity of the database:", SPECdbavg)
    return 0

def run_quality_metrics():
    meshes = dbmngr.get_all_with_extracted_features()

    if not meshes.alive:
        print("No meshes with extracted features found in the db. Run 'python main.py extract' first.")
        return

    dblabel = []
    for mesh in meshes:
        if "TruthTable" not in mesh:
            raise Exception("Error: No TruthTable in database. Please run getTruthTable() first.")
        dblabel.append(mesh['class'])

    # get all label in db
    dblabel = set(dblabel)
    DBlabel = []
    for label in dblabel:
        DBlabel.append(''.join(label))

    # Correct percentage of total data:
    #acc(DBlabel)

    # this one might be good to show
    # proportion of returned dogs from all RETURNED items
    # high precision(ppv): I get mostly dogs in my query result
    # low precision(ppv): I get many cats in my query result
    ppv(DBlabel)

    # proportion of returned dogs from all DOGS in database
    # high recall: I get most of dogs in database
    # low recall: There are many dogs in database I don’t find
    # also called sensitivity : proportion of all dogs that are returned by a query for dogs
    #recall(DBlabel)

    # this one has error
    # proportion of all cats that are not returned by a query for dogs
    #specificity(DBlabel)


if __name__ == "__main__":
    #getTruthTable()    #compute truthTable and store in db
    
    run_quality_metrics()

