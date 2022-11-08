import distance_functions as df
import database as db
import numpy as np

def analyze():
    dbmngr = db.DatabaseManager()

    query = dbmngr.query({'class': 'Bird'})[0]
    print ("Loaded query mesh: '" + query["path"] + "'")

    best = df.find_best_matches(query, k=10, verbose=False)

    for i, res in enumerate(best):
        tot = np.sum(np.asarray(res["distance"][1:]))

        print(" #" + str(i + 1) + " '" + res["path"] + "':")
        print("  Class: " + res["class"])
        print("  Total distance: " + str(res["distance"][0]) + " (" + str(res["distance"][0] * 100 / tot) + "%)")
        print("  Distance by feature:")
        print("  - surface_area: ", str(res["distance"][1]) + " (" + str(res["distance"][1] * 100 / tot) + "%)")
        print("  - compactness: ", str(res["distance"][2]) + " (" + str(res["distance"][2] * 100 / tot) + "%)")
        print("  - volume: ", str(res["distance"][3]) + " (" + str(res["distance"][3] * 100 / tot) + "%)")
        print("  - diameter: ", str(res["distance"][4]) + " (" + str(res["distance"][4] * 100 / tot) + "%)")
        print("  - eccentricity: ", str(res["distance"][5]) + " (" + str(res["distance"][5] * 100 / tot) + "%)")
        print("  - rectangularity: ", str(res["distance"][6]) + " (" + str(res["distance"][6] * 100 / tot) + "%)")
        print("  - A3: ", str(res["distance"][7]) + " (" + str(res["distance"][7] * 100 / tot) + "%)")
        print("  - D1: ", str(res["distance"][8]) + " (" + str(res["distance"][8] * 100 / tot) + "%)")
        print("  - D2: ", str(res["distance"][9]) + " (" + str(res["distance"][9] * 100 / tot) + "%)")
        print("  - D3: ", str(res["distance"][10]) + " (" + str(res["distance"][10] * 100 / tot) + "%)")
        print("  - D4: ", str(res["distance"][11]) + " (" + str(res["distance"][11] * 100 / tot) + "%)")

def simmulated_annealing():
    dbmngr = db.DatabaseManager()

    query = dbmngr.query({'class': 'Bird'})[0]
    print ("Loaded query mesh: '" + query["path"] + "'")

    iterations = 500
    weights = [-317.79075611632203, -1303.6812032556472, -62.687294985121106, -152.70488648079944, -2766.319887351074, -58.33360130866355, -10.858404101835053, -17.586064890370473, -10.606783009335377, -28.35758719917543, -19.02277353030694]
    weights = np.asarray(weights)
    score = -1
    lastScore = -1

    for it in range(iterations):
        best = df.find_best_matches(query, k=10, verbose=False, weights=weights)
        lastScore = score
        score = 0

        for i, res in enumerate(best):
            tot = np.sum(np.asarray(res["distance"][1:]))
            res_weights = np.asarray([res["distance"][i] / tot for i in range(1, len(res["distance"]))])

            if res["class"] == query["class"]:
                # Reward
                weights += res_weights * 0.1
                score += 1
            else:
                # Punish
                weights -= res_weights * 0.1
                score -= 1
        
        print("Iteration #" + str(it + 1) + ": " + str(score) + " (" + str(score - lastScore) + ")")
        print("Weights: " + str(weights))

if __name__ == "__main__":
    # analyze()
    simmulated_annealing()