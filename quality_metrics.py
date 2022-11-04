from tqdm import tqdm
import database as db
import distance_functions as df

dbmngr = db.DatabaseManager()

def getTruthTable():
    meshes = dbmngr.get_all_with_extracted_features()
    # mesh = dbmngr.get_by_path("data\\normalized\\Airplane\\61.off")
    for mesh in tqdm(meshes, desc='Computing TruthTable', ncols=130, total=dbmngr.get_mesh_count()):
        Qlabel = mesh['class']
        res = df.find_best_matches(mesh, k=5)
        TP = FP = 0
        # dbQsum: total number of positive shapes in db
        dbQlabel = dbmngr.get_all_by_category(Qlabel)
        dbQsum = 0
        for n in dbQlabel:
            dbQsum += 1

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
                dbRlabel = dbmngr.get_all_by_category(r['class'])
                for n in dbRlabel:
                    dbRsum += 1
        TN = dbRsum - FP

        TruthTable = [TP, FP, FN, TN]
        mesh["TruthTable"] = TruthTable
        dbmngr.update_one(mesh)
    return 0


if __name__ == "__main__":
    getTruthTable()