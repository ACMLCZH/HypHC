import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.metrics import accuracy_score

from datasets.loading import load_data


def dendrogram_purity(X, y, linkage_method='ward'):
    Z = linkage(X, method=linkage_method)
    ddata = dendrogram(Z, no_plot=True)
    n_leaves = len(ddata['leaves'])
    n_clusters = n_leaves
    purity_scores = []
    for i in range(n_leaves - 1):
        n_clusters -= 1
        cluster_idxs = ddata['color_list'][:n_clusters]
        cluster_labels = [y[ddata['leaves'][cluster_idxs == i]] for i in range(n_clusters)]
        majority_labels = [max(set(labels), key=list(labels).count) for labels in cluster_labels]
        purity = accuracy_score(y, majority_labels)
        purity_scores.append(purity)
    return purity_scores


def umpga(args):
    x, y_true, similarities = load_data(args.dataset)
    
    n = similarities.shape[0]
    
    for
    #     public static List<int[]> UPGMA(double[,] dist, int numTree, string method)
    #     {
    #         int n = dist.GetLength(0);
    #         PriorityQueue<(int, int), double> pq = new();
    #         for (int i = 0; i < n; ++i)
    #             for (int j = 0; j < i; ++j)
    #                 pq.Enqueue((i, j), dist[i, j]);

    #         List<bool> merged = new(Enumerable.Repeat(false, n));
    #         List<List<int>> leaves = new(
    #             Enumerable.Range(0, n).Select(i => new List<int>(new int[] { i }))
    #         );
    #         List<int[]> edges = new();
    #         for (int ty = 0; ty < n - numTree; ++ty)
    #         {
    #             (int i, int j) curPair;
    #             double curDist;
    #             do pq.TryDequeue(out curPair, out curDist);
    #                 while (merged[curPair.i] || merged[curPair.j]);

    #             Console.WriteLine($"{curPair} {curDist}");

    #             int newNode = n + ty;
    #             merged[curPair.i] = true;
    #             merged[curPair.j] = true;
    #             merged.Add(false);
    #             leaves.Add(new List<int>(leaves[curPair.i].Concat(leaves[curPair.j])));
    #             edges.Add(new int[] { curPair.i, newNode });
    #             edges.Add(new int[] { curPair.j, newNode });

    #             for (int ot = 0; ot < newNode; ++ot)
    #                 if (!merged[ot])
    #                 {
    #                     double totDis = 0.0;
    #                     foreach (int i in leaves[ot])
    #                         foreach (int j in leaves[newNode])
    #                             totDis += dist[i, j];
    #                     pq.Enqueue((ot, newNode), totDis / (leaves[ot].Count * leaves[newNode].Count));
    #                 }
    #         }
    #         //foreach (var edge in edges)
    #         //    Console.WriteLine($"{edge[0]}, {edge[1]}");

    #         return edges;
    #     }
	# }