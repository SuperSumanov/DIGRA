#ifndef RANGEHNSW_RANGEHNSW_HPP
#define RANGEHNSW_RANGEHNSW_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <limits>
#include <cassert>

#include "hnswlib/hnswlib.h"
#define BTREE_M 3
#define BTREE_D 2


using namespace hnswlib;

#include <sys/resource.h>
#include <unistd.h>

class RangeHNSW {
public:
    RangeHNSW(
            int d,
            size_t eleNum,
            size_t maxEleNum,
            float* vecData,
            int* keyList,
            int* valueList,
            int m,
            int ef_con
    ):
            M(m),ef_construction(ef_con), space(d), dim(d), linklist(maxEleNum), searchLayer(maxEleNum), eleCount(eleNum), maxNum(maxEleNum){

        skipLayer = log(M)/log(BTREE_D);
        maxLayer = floor(log((float)maxEleNum) / log(BTREE_D));

        // Cluster size ~ logN for O(n) memory
        clusterSize = std::max(2, (int)floor(log2((double)eleNum)));

        visited_array = new unsigned int[maxEleNum];

        std::random_device rd;
        eng = std::mt19937 (rd());

        data_size_ = space.get_data_size();
        fstdistfunc_ = space.get_dist_func();
        dist_func_param_ = space.get_dist_func_param();

        keyList_ = new int[maxEleNum];
        valueList_ = new int[maxEleNum];
        vecData_ = (char*)(new float[maxEleNum * dim]);
        isDeleted = new bool[maxEleNum];
        memset(isDeleted,0,maxEleNum);
        memcpy(keyList_,keyList, eleNum * sizeof(int));
        memcpy(valueList_,valueList, eleNum * sizeof(int));
        memcpy(vecData_,vecData, eleNum * dim * sizeof(float));

        mult_ = 1 / log(1.0 * M);
        revSize_ = 1.0 / mult_;

        sizeLinkList = (M * sizeof(tableint) + sizeof(linklistsizeint));

        space = hnswlib::L2Space(dim);

        sortedArray.reserve(eleNum);

        // Allocate layer-0 only per-point linklist: O(n) instead of O(n*logn)
        for(int i = 0; i < (int)eleNum; i++){
            key2Id[keyList[i]] = i;
            linklist[i] = (char *) malloc(sizeLinkList);
            sortedArray.push_back(i);
        }

        // Initialize cluster layers (one per layer >= 1)
        clusterLayers.resize(maxLayer);
        for(int l = 0; l < maxLayer; l++){
            clusterLayers[l].pointToCluster.assign(maxEleNum, -1);
        }

        sort(sortedArray.begin(),sortedArray.end(),[this](int a, int b) { return this->cmp(a, b); });

        root = buildTree(eleNum);

    }

    std::priority_queue<std::pair<float, hnswlib::labeltype>> queryRange(float *vecData, int rangeL, int rangeR, int k,int ef_s){
        node* highNode = findHighNode(root,rangeL,rangeR);

        int belongL = highNode->keynum;
        int belongR = highNode->keynum;
        for(int i = 0 ; i < highNode->keynum; i++){
            if(rangeL < valueList_[highNode->key[i]] ||
               (rangeL == valueList_[highNode->key[i]] && (rangeL == valueList_[findRight(highNode->child[i])]))){
                belongL = i;
                break;
            }
        }
        for(int i = 0 ; i < highNode->keynum; i++){
            if(rangeR < valueList_[highNode->key[i]]){
                belongR = i;
                break;
            }
        }
        std::vector<tableint > ep_ids;
        std::priority_queue<std::pair<float, hnswlib::labeltype>> top;
        if(belongL == belongR) {
            top.push({0,keyList_[highNode->entryPoint]});
            return top;
        }
        int sp;
        if(belongL == belongR - 1){
            node* nodeL = highNode->child[belongL];
            while(nodeL->layer != 0 && valueList_[nodeL->key[nodeL->keynum - 1]] < rangeL) nodeL = nodeL->child[nodeL->keynum];
            tableint ep1 = nodeL->layer != 0 ? findEntry(vecData,nodeL,nodeL->child[nodeL->keynum]->entryPoint) : nodeL->entryPoint;
            ep_ids.push_back(ep1);
            searchLayer[ep1] = nodeL->layer;
            sp = highNode->key[belongL];

            node* nodeR = highNode->child[belongR];
            while(nodeR->layer != 0 && valueList_[nodeR->key[0]] > rangeR) nodeR = nodeR->child[0];
            tableint ep2 = nodeR->layer != 0 ? findEntry(vecData,nodeR,nodeR->child[0]->entryPoint) : nodeR->entryPoint;
            ep_ids.push_back(ep2);
            searchLayer[ep2] = nodeR->layer;
        }
        else{
            sp = -1;
            std::uniform_int_distribution<> distr(belongL + 1, belongR -1);
            tableint high_ep = highNode->child[distr(eng)]->entryPoint;
            tableint ep = findEntry(vecData,highNode, high_ep);
            ep_ids.push_back(ep);
            searchLayer[ep] = highNode->layer;
        }
        ResultHeap result = searchBaseLayer0(ep_ids,vecData,highNode->layer,rangeL,rangeR,ef_s,sp);

        while(result.size() > k) result.pop();

        while(!result.empty()){
            auto r = result.top();
            result.pop();
            top.push({r.first,r.second});
        }
        return top;
    }

    void addPoint(int key,int value, char* data){
        keyList_[eleCount] = key;
        key2Id[key] = eleCount;
        valueList_[eleCount] = value;
        memcpy(vecData_+ dim * sizeof(float) * eleCount, data, dim * sizeof(float));
        // Allocate layer-0 only linklist
        linklist[eleCount] = (char *) malloc(sizeLinkList);
        unsigned int *newListData = (unsigned int *) get_linklist(eleCount, 0);
        setListCount(newListData, 0);
        eleCount ++;
        addPoint(eleCount - 1);
    }

    void erase(int key){
        int id = key2Id[key];
        isDeleted[id] = true;
        erase(root,id);
        if(root->keynum == 0) root = root->child[0];
    }

    void resize(size_t newMaxN){
        int maxEleNum = newMaxN;
        skipLayer = log(M)/log(BTREE_D);

        maxLayer = floor(log((float)maxEleNum) / log(BTREE_D));

        visited_array = new unsigned int[maxEleNum];

        std::random_device rd;
        eng = std::mt19937 (rd());

        data_size_ = space.get_data_size();
        fstdistfunc_ = space.get_dist_func();
        dist_func_param_ = space.get_dist_func_param();

        keyList_ = (int*) realloc(keyList_, maxEleNum * sizeof(int));
        valueList_ = (int*) realloc(valueList_, maxEleNum * sizeof(int));
        vecData_ = (char*)realloc(vecData_,maxEleNum * dim * sizeof(float ));
        isDeleted = (bool*) realloc(isDeleted, maxEleNum * sizeof(bool));
        memset(isDeleted,0,maxEleNum);

        mult_ = 1 / log(1.0 * M);
        revSize_ = 1.0 / mult_;

        sizeLinkList = (M * sizeof(tableint) + sizeof(linklistsizeint));

        space = hnswlib::L2Space(dim);
        linklist.resize(maxEleNum);

        // Layer-0 only realloc
        for(int i = 0; i < (int)maxNum; i++){
            linklist[i] = (char *) realloc(linklist[i], sizeLinkList);
        }
        for(int i = maxNum; i < maxEleNum; i++){
            linklist[i] = (char *) malloc(sizeLinkList);
        }

        // Resize cluster layers
        clusterSize = std::max(2, (int)floor(log2((double)maxEleNum)));
        clusterLayers.resize(maxLayer);
        for(int l = 0; l < maxLayer; l++){
            clusterLayers[l].pointToCluster.resize(maxEleNum, -1);
        }
    }

private:

    size_t maxNum, eleCount;
    size_t sizeLinkList;
    struct node{
        int entryPoint = -1;
        int keynum = 0;
        int key[BTREE_M];
        struct node* child[BTREE_M + 1];
        short int layer; //layer in tree

        node(){}

    };

    bool cmp(int a,int b){
        if(valueList_[a]!=valueList_[b]) return valueList_[a]<valueList_[b];
        else return keyList_[a]<keyList_[b];
    }

    struct CompareByFirst {
        constexpr bool operator()(std::pair<float, tableint> const& a,
                                  std::pair<float, tableint> const& b) const noexcept {
            return a.first < b.first;
        }
    };

    typedef std::priority_queue<std::pair<float, tableint>, std::vector<std::pair<float , tableint>>, CompareByFirst> ResultHeap;

    hnswlib::L2Space space;
    size_t data_size_{0};

    DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_{nullptr};

    node* root;
    char* vecData_;
    int* keyList_;
    int* valueList_;
    bool* isDeleted;
    int threshold;
    int M,M0;
    int skipLayer = 1;
    int ef_construction;
    int dim;

    int numEdges = 0;


    float alpha;
    double mult_{0.0}, revSize_{0.0};

    int maxLayer;

    std::vector<char *> linklist;

    std::vector<short int> searchLayer;
    unsigned int *visited_array;
    unsigned int tag = 0;
    std::vector<int> sortedArray;

    std::unordered_map<int,int> key2Id;

    std::mt19937 eng;

    // -------------------------------------------------------
    // KMeans cluster optimization data structures
    // -------------------------------------------------------
    int clusterSize = 2; // target cluster size ~ logN

    struct ClusterLayer {
        std::vector<int> pointToCluster;        // pointToCluster[pointId] = global cluster ID
        std::vector<tableint> repPoint;         // repPoint[clusterId] = representative point
        std::vector<std::vector<tableint>> members; // members[clusterId] = list of point IDs
        std::vector<char*> edgeList;            // edgeList[clusterId] = HNSW edges (sizeLinkList bytes)
        int numClusters = 0;
    };

    std::vector<ClusterLayer> clusterLayers; // clusterLayers[l-1] for layer l >= 1

    // KMeans clustering using L2 distance on the actual vectors
    // pointIds: point IDs to cluster; K: target number of clusters
    // Returns assignments[i] = local cluster ID for pointIds[i]
    void kmeansL2(const std::vector<tableint>& pointIds, int K, int maxIter,
                  std::vector<int>& assignments, std::vector<std::vector<float>>& centroids) {
        int n = (int)pointIds.size();
        if(K <= 0) K = 1;
        if(K > n) K = n;

        assignments.resize(n);
        centroids.resize(K, std::vector<float>(dim, 0.0f));

        // Initialize centroids: pick K distinct random points
        std::vector<int> indices(n);
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), eng);

        for(int k = 0; k < K; k++){
            float* vec = (float*)getDataByInternalId(pointIds[indices[k]]);
            for(int d = 0; d < dim; d++) centroids[k][d] = vec[d];
        }

        std::vector<int> counts(K);
        for(int iter = 0; iter < maxIter; iter++){
            bool changed = false;
            // Assign each point to nearest centroid
            for(int i = 0; i < n; i++){
                float* vec = (float*)getDataByInternalId(pointIds[i]);
                float bestDist = std::numeric_limits<float>::max();
                int bestK = 0;
                for(int k = 0; k < K; k++){
                    float dist = 0;
                    for(int d = 0; d < dim; d++){
                        float diff = vec[d] - centroids[k][d];
                        dist += diff * diff;
                    }
                    if(dist < bestDist){ bestDist = dist; bestK = k; }
                }
                if(assignments[i] != bestK) changed = true;
                assignments[i] = bestK;
            }
            if(!changed) break;

            // Update centroids
            for(int k = 0; k < K; k++){
                counts[k] = 0;
                for(int d = 0; d < dim; d++) centroids[k][d] = 0;
            }
            for(int i = 0; i < n; i++){
                int k = assignments[i];
                counts[k]++;
                float* vec = (float*)getDataByInternalId(pointIds[i]);
                for(int d = 0; d < dim; d++) centroids[k][d] += vec[d];
            }
            for(int k = 0; k < K; k++){
                if(counts[k] > 0){
                    float inv = 1.0f / counts[k];
                    for(int d = 0; d < dim; d++) centroids[k][d] *= inv;
                }
            }
        }
    }

    // Assign a new point to the nearest existing cluster at a given layer
    void assignToNearestCluster(tableint pointId, int layer) {
        ClusterLayer& cl = clusterLayers[layer - 1];
        float* pvec = (float*)getDataByInternalId(pointId);
        float bestDist = std::numeric_limits<float>::max();
        int bestCluster = 0;
        for(int c = 0; c < cl.numClusters; c++){
            float dist = 0;
            float* repVec = (float*)getDataByInternalId(cl.repPoint[c]);
            for(int d = 0; d < dim; d++){
                float diff = pvec[d] - repVec[d];
                dist += diff * diff;
            }
            if(dist < bestDist){ bestDist = dist; bestCluster = c; }
        }
        cl.pointToCluster[pointId] = bestCluster;
        cl.members[bestCluster].push_back(pointId);
    }

    int findEntryLayer(int Layer) const{
        return Layer % skipLayer;
    }

    int findRight(node* nd){
        if(nd->layer == 0) return nd->entryPoint;
        else return findRight(nd->child[nd->keynum]);
    }

    void updateEntry(node *nd){
        std::uniform_int_distribution<> distr(0,  nd->keynum);
        nd->entryPoint = nd->child[distr(eng)]->entryPoint;
    }

    node* buildTree(int eleNum){
        std::queue<std::pair< std::pair<int,int>, node* > > q[2];
        int qid = 0;
        for(int i = 0; i < eleNum; i++){
            node *nd = new node();
            nd->layer = 0;
            nd->entryPoint = sortedArray[i];
            q[qid].push({{i, i}, nd});
            unsigned int *newListData = (unsigned int *) get_linklist(sortedArray[i], 0);

            setListCount(newListData, 0);

        }
        while(q[qid].size() > 1){
            std::cout<<"layer:"<<q[qid].front().second->layer<<std::endl;
            int nxtqid = qid ^ 1;
            while(!q[qid].empty()){
                std::vector<std::pair<int,int>> tmp;

                int numChild;
                if(q[nxtqid].size()%2 == 0){
                    numChild = (q[qid].size() >= 2 * BTREE_D) ? BTREE_D : q[qid].size();
                }
                else {
                    if (q[qid].size() >= 2 * BTREE_M) numChild = BTREE_M;
                    else if (q[qid].size() <= BTREE_M) numChild = q[qid].size();
                    else numChild = q[qid].size() / 2;
                }
                tmp.reserve(numChild);
                tmp.resize(numChild);
                node* nd = new node();
                nd->keynum = numChild - 1 ;
                for(int i = 0; i < numChild; i++){
                    auto t = q[qid].front();
                    tmp[i] = t.first;
                    q[qid].pop();
                    if (i != 0){
                        nd->key[i - 1] = sortedArray[tmp[i].first];
                    }
                    nd->child[i] = t.second;
                }
                std::uniform_int_distribution<> distr(0, numChild - 1);
                nd->entryPoint = nd->child[distr(eng)]->entryPoint;
                int layer = nd->layer = nd->child[0]->layer + 1;

                // --- KMeans clustering: cluster all points in this tree node ---
                // Collect all point IDs in this tree node
                std::vector<tableint> allPoints;
                for(int i = 0; i < numChild; i++){
                    for(int ii = tmp[i].first; ii <= tmp[i].second; ii++){
                        allPoints.push_back(sortedArray[ii]);
                    }
                }

                int nPts = (int)allPoints.size();
                int K = std::max(1, (int)ceil((double)nPts / clusterSize));

                // KMeans cluster using L2 distance
                std::vector<int> assignments;
                std::vector<std::vector<float>> centroids;
                kmeansL2(allPoints, K, 15, assignments, centroids);

                // Group points by cluster
                std::vector<std::vector<tableint>> localMembers(K);
                for(int i = 0; i < nPts; i++){
                    localMembers[assignments[i]].push_back(allPoints[i]);
                }

                // Find representative (nearest to centroid) for each cluster
                std::vector<tableint> reps(K);
                for(int k = 0; k < K; k++){
                    float bestDist = std::numeric_limits<float>::max();
                    reps[k] = localMembers[k].empty() ? allPoints[0] : localMembers[k][0];
                    for(tableint pid : localMembers[k]){
                        float* vec = (float*)getDataByInternalId(pid);
                        float dist = 0;
                        for(int d = 0; d < dim; d++){
                            float diff = vec[d] - centroids[k][d];
                            dist += diff * diff;
                        }
                        if(dist < bestDist){ bestDist = dist; reps[k] = pid; }
                    }
                }

                // Build a mapping from point to child index for this tree node
                std::unordered_map<tableint, int> pointChildMap;
                for(int i = 0; i < numChild; i++){
                    for(int ii = tmp[i].first; ii <= tmp[i].second; ii++){
                        pointChildMap[sortedArray[ii]] = i;
                    }
                }

                // Register clusters globally in clusterLayers[layer-1]
                ClusterLayer& cl = clusterLayers[layer - 1];
                int baseClusterId = cl.numClusters;

                for(int k = 0; k < K; k++){
                    int globalCId = baseClusterId + k;
                    cl.repPoint.push_back(reps[k]);
                    cl.members.push_back(localMembers[k]);

                    // Allocate cluster edge list
                    char* edgeMem = (char*)malloc(sizeLinkList);
                    setListCount((linklistsizeint*)edgeMem, 0);
                    cl.edgeList.push_back(edgeMem);

                    // Map all members to this global cluster
                    for(tableint pid : localMembers[k]){
                        cl.pointToCluster[pid] = globalCId;
                    }
                }
                cl.numClusters += K;

                // Build HNSW edges between cluster representatives
                for(int k = 0; k < K; k++){
                    tableint repId = reps[k];
                    char *data = getDataByInternalId(repId);
                    ResultHeap candidates;

                    // Seed with lower-layer edges of the representative
                    unsigned int *listData = (unsigned int *) get_linklist(repId, layer - 1);
                    int size = getListCount(listData);
                    tableint *listD = (tableint *) (listData + 1);
                    for(int j = 0; j < size; j++){
                        candidates.emplace(
                            fstdistfunc_(data, getDataByInternalId(listD[j]),
                                         dist_func_param_), listD[j]);
                    }

                    // Search sibling subtrees for more neighbors
                    int repChild = pointChildMap[repId];
                    for(int j = 0; j < numChild; j++){
                        if(j != repChild){
                            tableint ep_id = findEntry(data, nd->child[j], nd->child[j]->entryPoint);
                            std::vector<tableint> ep_ids = {ep_id};
                            ResultHeap r = searchBaseLayer(ep_ids, data, layer - 1);
                            getNeighborsByHeuristic2(r, M);
                            while(!r.empty()){
                                candidates.push(r.top());
                                r.pop();
                            }
                        }
                    }

                    // Also add other cluster representatives as candidates
                    for(int k2 = 0; k2 < K; k2++){
                        if(k2 != k){
                            float dist = fstdistfunc_(data, getDataByInternalId(reps[k2]), dist_func_param_);
                            candidates.emplace(dist, reps[k2]);
                        }
                    }

                    getNeighborsByHeuristic2(candidates, M);

                    // Store edges in cluster's edge list
                    int globalCId = baseClusterId + k;
                    unsigned int *newListData = (unsigned int *) cl.edgeList[globalCId];
                    tableint *newListD = (tableint *) (newListData + 1);
                    int indx = 0;
                    while(candidates.size() > 0){
                        newListD[indx] = candidates.top().second;
                        candidates.pop();
                        indx++;
                    }
                    setListCount(newListData, indx);
                    numEdges += indx;
                }

                q[nxtqid].push({{tmp[0].first,tmp[tmp.size() - 1].second}, nd});
            }

            qid = nxtqid;
        }
        std::cout<<"edge num:"<<numEdges<<std::endl;
        if(!clusterLayers.empty() && clusterLayers[0].numClusters > 0)
            std::cout<<"cluster edges (avg per cluster):"<<numEdges*1.0/clusterLayers[0].numClusters<<std::endl;
        std::cout<<"cluster size target:"<<clusterSize<<std::endl;
        return q[qid].front().second;
    }

    void traverse(std::vector<tableint > &result,node* nd)
    {
        if (nd->layer == 0){
            result.push_back(nd->entryPoint);
            return;
        }
        for(int i = 0; i <= nd->keynum; i++) traverse(result,nd->child[i]);
    }


    void addPoint(int id){
        if (root == NULL)
        {
            root = new node();
            root->keynum = 1;
        }
        else
        {
            insert(root, id);
            if(root->keynum == BTREE_M){
                node *newRoot = new node();
                newRoot->layer = root->layer + 1;
                newRoot->keynum = 0;
                newRoot->child[0] = root;
                root = newRoot;
                // With clustering, higher-layer edges are cluster-level.
                // Set up cluster for new top layer
                if(newRoot->layer <= maxLayer && newRoot->layer >= 1){
                    ClusterLayer& clNew = clusterLayers[newRoot->layer - 1];
                    // Assign all points to a single cluster at new top layer
                    for(int i = 0; i < (int)eleCount; i++){
                        clNew.pointToCluster[i] = 0;
                    }
                    if(clNew.numClusters == 0){
                        clNew.repPoint.push_back(root->entryPoint);
                        std::vector<tableint> allPts;
                        for(int i = 0; i < (int)eleCount; i++) allPts.push_back(i);
                        clNew.members.push_back(allPts);
                        char* edgeMem = (char*)malloc(sizeLinkList);
                        setListCount((linklistsizeint*)edgeMem, 0);
                        clNew.edgeList.push_back(edgeMem);
                        clNew.numClusters = 1;
                    }
                }
                splitNode(newRoot,0);
            }
        }
    }



    void erase(node *nd, int id){
        int belong = nd->keynum;
        for(int i = 0 ; i < nd->keynum; i++){
            if( cmp(id, nd->key[i])){
                belong = i;
                break;
            }
        }
        if(nd->layer == 1){

            for(int i = std::max(0,belong -1); i < nd->keynum - 1; i ++){
                nd->key[i] = nd->key[i + 1];
            }
            for(int i = belong; i <= nd->keynum - 1; i ++){
                nd->child[i] = nd->child[i +1];
            }
            nd->keynum --;
            return;
        }
        else {
            erase(nd->child[belong], id);
            if (nd->child[belong]->keynum < BTREE_D - 1) {
                node *nd1 = nd->child[belong];
                if(belong > 0 && nd->child[belong - 1]->keynum >= BTREE_D){
                    node *nd2 = nd->child[belong-1];
                    for(int i = nd1->keynum - 1; i >= 0; i--){
                        nd1->key[i + 1] = nd1->key[i];
                    }
                    for(int i = nd1->keynum; i >= 0; i--){
                        nd1->child[i + 1] = nd1->child[i];
                    }
                    nd1->key[0] = nd->key[belong - 1];

                    nd1->child[0] = nd2->child[nd2->keynum];
                    nd1->keynum ++;

                    nd->key[belong - 1] = nd2->key[nd2->keynum - 1];
                    nd2->keynum --;

                    refresh(nd1, 0);
                    refresh(nd2);
                }
                else if(belong < nd->keynum && nd->child[belong + 1]->keynum >= BTREE_D){
                    node *nd2 = nd->child[belong + 1];

                    nd1->key[nd1->keynum] = nd->key[belong];
                    nd1->keynum ++;
                    nd1->child[nd1->keynum] = nd2->child[0];

                    nd->key[belong] = nd2->key[0];
                    for(int i = 0 ; i < nd2->keynum ; i++){
                        nd2->key[i] = nd2->key[i + 1];
                    }
                    for(int i = 0 ; i <= nd2->keynum ; i++){
                        nd2->child[i] = nd2->child[i + 1];
                    }
                    nd2->keynum --;

                    refresh(nd1, nd1->keynum);

                    refresh(nd2);
                }
                else if(belong > 0){
                    mergeNode(nd, belong - 1);
                }
                else{
                    mergeNode(nd, belong);
                }
            }
        }
    }

    void refresh(node *nd, int refreshId){
        // With clustering: re-cluster the subtree and rebuild cluster edges
        refreshClusterEdges(nd);
    }

    void mergeNode(node *nd, int mergeId){
        node* n1 = nd->child[mergeId];
        node* n2 = nd->child[mergeId + 1];
        n1->key[n1->keynum] = nd->key[mergeId];
        for(int i = 0; i < n2->keynum; i++){
            n1->key[i + n1->keynum + 1] = n2->key[i];
        }
        for(int i = 0; i <= n2->keynum; i++){
            n1->child[i + n1->keynum + 1] = n2->child[i];
        }
        n1->keynum += n2->keynum + 1;
        refresh(n1);
        for(int i = mergeId; i < nd->keynum; i++){
            nd->key[i] = nd->key[i + 1];
        }
        for(int i = mergeId + 1; i <= nd->keynum; i++){
            nd->child[i] = nd->child[i + 1];
        }
        nd->keynum --;
        updateEntry(nd);
    }

    tableint insert(node* nd, int id){
        int belong = nd->keynum;
        for(int i = 0 ; i < nd->keynum; i++){
            if( cmp(id, nd->key[i])){
                belong = i;
                break;
            }
        }
        tableint ep_id;
        if(nd->layer == 1){
            node *newnd = new node();
            newnd->layer = 0;
            newnd->entryPoint = id;

            // Insert into B-tree at layer 1
            for(int i = nd->keynum -1; i >= belong; i --){
                nd->key[i + 1] = nd->key[i];
            }
            for(int i = nd->keynum; i > belong; i --){
                nd->child[i + 1] = nd->child[i];
            }
            if(belong == 0){
                if(cmp(id,nd->child[0]->entryPoint)){
                    std::swap(nd->child[0],newnd);
                }
            }
            nd->key[belong] = newnd->entryPoint;
            nd->keynum ++;
            nd->child[belong + 1] = newnd;
            ep_id = nd->entryPoint;
        }
        else {
            ep_id = insert(nd->child[belong], id);
            if (nd->child[belong]->keynum == BTREE_M) {
                splitNode(nd, belong);
            }
        }

        // For all layers >= 1: assign to nearest cluster instead of per-point edges
        if(nd->layer >= 1 && nd->layer <= maxLayer){
            assignToNearestCluster(id, nd->layer);
            ClusterLayer& cl = clusterLayers[nd->layer - 1];
            int clId = cl.pointToCluster[id];
            return cl.repPoint[clId];
        }

        // Fallback (shouldn't reach here for this tree structure)
        return ep_id;
    }

    void splitNode(node *nd, int splitId){
        node* n1 = nd->child[splitId];
        node* n2 = new node();
        n2->layer = n1->layer;
        n2->keynum = 0;
        int splitPoint = n1->keynum/2;
        int newKey = n1->key[splitPoint];
        for(int i = splitPoint + 1; i < n1->keynum; i++){
            n2->key[i-splitPoint - 1] = n1->key[i];
        }
        for(int i = splitPoint + 1; i <= n1->keynum; i++){
            n2->child[i-splitPoint - 1] = n1->child[i];
        }
        n2->keynum = n1->keynum - splitPoint - 1;
        n1->keynum = splitPoint;
        refresh(n1);
        refresh(n2);
        for(int i = nd->keynum -1; i >= splitId; i --){
            nd->key[i + 1] = nd->key[i];
        }
        for(int i = nd->keynum; i > splitId; i --){
            nd->child[i + 1] = nd->child[i];
        }
        nd->keynum ++;
        nd->key[splitId] = newKey;
        nd->child[splitId + 1] = n2;
        updateEntry(nd);
    }


    void refresh(node *nd){
        // With clustering: re-cluster the subtree and rebuild cluster edges
        refreshClusterEdges(nd);
    }

    // Re-cluster and rebuild cluster edges for a tree node's subtree
    void refreshClusterEdges(node* nd){
        if(nd->layer < 1 || nd->layer > maxLayer) return;
        int layer = nd->layer;
        ClusterLayer& cl = clusterLayers[layer - 1];

        // Collect all points in the subtree
        std::vector<tableint> allPoints;
        traverse(allPoints, nd);

        int nPts = (int)allPoints.size();
        if(nPts == 0) return;
        int K = std::max(1, (int)ceil((double)nPts / clusterSize));

        // KMeans cluster
        std::vector<int> assignments;
        std::vector<std::vector<float>> centroids;
        kmeansL2(allPoints, K, 15, assignments, centroids);

        // Group points by cluster
        std::vector<std::vector<tableint>> localMembers(K);
        for(int i = 0; i < nPts; i++){
            localMembers[assignments[i]].push_back(allPoints[i]);
        }

        // Find representatives
        std::vector<tableint> reps(K);
        for(int k = 0; k < K; k++){
            float bestDist = std::numeric_limits<float>::max();
            reps[k] = localMembers[k].empty() ? allPoints[0] : localMembers[k][0];
            for(tableint pid : localMembers[k]){
                float* vec = (float*)getDataByInternalId(pid);
                float dist = 0;
                for(int d = 0; d < dim; d++){
                    float diff = vec[d] - centroids[k][d];
                    dist += diff * diff;
                }
                if(dist < bestDist){ bestDist = dist; reps[k] = pid; }
            }
        }

        // Register clusters
        int baseClusterId = cl.numClusters;
        for(int k = 0; k < K; k++){
            int globalCId = baseClusterId + k;
            cl.repPoint.push_back(reps[k]);
            cl.members.push_back(localMembers[k]);
            char* edgeMem = (char*)malloc(sizeLinkList);
            setListCount((linklistsizeint*)edgeMem, 0);
            cl.edgeList.push_back(edgeMem);
            for(tableint pid : localMembers[k]){
                cl.pointToCluster[pid] = globalCId;
            }
        }
        cl.numClusters += K;

        // Build cluster edges using representatives
        for(int k = 0; k < K; k++){
            tableint repId = reps[k];
            char *data = getDataByInternalId(repId);
            ResultHeap candidates;

            unsigned int *listData = (unsigned int *) get_linklist(repId, layer - 1);
            int size = getListCount(listData);
            tableint *listD = (tableint *) (listData + 1);
            for(int j = 0; j < size; j++){
                if(!isDeleted[listD[j]])
                    candidates.emplace(fstdistfunc_(data, getDataByInternalId(listD[j]),
                                                     dist_func_param_), listD[j]);
            }

            for(int j = 0; j <= nd->keynum; j++){
                tableint ep_id = findEntry(data, nd->child[j], nd->child[j]->entryPoint);
                std::vector<tableint> ep_ids = {ep_id};
                ResultHeap r = searchBaseLayer(ep_ids, data, layer - 1);
                getNeighborsByHeuristic2(r, M);
                while(!r.empty()){ candidates.push(r.top()); r.pop(); }
            }

            for(int k2 = 0; k2 < K; k2++){
                if(k2 != k){
                    float dist = fstdistfunc_(data, getDataByInternalId(reps[k2]), dist_func_param_);
                    candidates.emplace(dist, reps[k2]);
                }
            }

            getNeighborsByHeuristic2(candidates, M);

            int globalCId = baseClusterId + k;
            unsigned int *newListData = (unsigned int *) cl.edgeList[globalCId];
            tableint *newListD = (tableint *) (newListData + 1);
            int indx = 0;
            while(candidates.size() > 0){
                newListD[indx] = candidates.top().second;
                candidates.pop();
                indx++;
            }
            setListCount(newListData, indx);
        }
        updateEntry(nd);
    }

    tableint connectEdges(
            const void *data_point,
            tableint cur_c,
            ResultHeap &top_candidates,
            int layer) {

        std::vector<tableint> selectedNeighbors;
        selectedNeighbors.reserve(M);
        while (top_candidates.size() > 0) {
            selectedNeighbors.push_back(top_candidates.top().second);
            top_candidates.pop();
        }

        tableint next_closest_entry_point = selectedNeighbors.back();

        {
            linklistsizeint *ll_cur = get_linklist(cur_c, layer);

            setListCount(ll_cur, selectedNeighbors.size());
            tableint *data = (tableint *) (ll_cur + 1);
            for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {
                data[idx] = selectedNeighbors[idx];
            }
        }

        for (size_t idx = 0; idx < selectedNeighbors.size(); idx++) {

            linklistsizeint *ll_other = get_linklist(selectedNeighbors[idx], layer);

            size_t sz_link_list_other = getListCount(ll_other);

            tableint *data = (tableint *) (ll_other + 1);
            if (sz_link_list_other < M) {
                data[sz_link_list_other] = cur_c;
                setListCount(ll_other, sz_link_list_other + 1);
            } else {
                // finding the "weakest" element to replace it with the new one
                float d_max = fstdistfunc_(getDataByInternalId(cur_c), getDataByInternalId(selectedNeighbors[idx]),
                                           dist_func_param_);
                // Heuristic:
                ResultHeap candidates;
                candidates.emplace(d_max, cur_c);

                for (size_t j = 0; j < sz_link_list_other; j++) {
                    candidates.emplace(
                            fstdistfunc_(getDataByInternalId(data[j]), getDataByInternalId(selectedNeighbors[idx]),
                                         dist_func_param_), data[j]);
                }

                getNeighborsByHeuristic2(candidates, M);

                int indx = 0;
                while (candidates.size() > 0) {
                    data[indx] = candidates.top().second;
                    candidates.pop();
                    indx++;
                }

                setListCount(ll_other, indx);
            }
        }

        return next_closest_entry_point;
    }

    node* findHighNode(node* node,int rangeL, int rangeR){
        if(node->layer == 0){
            return node;
        }
        int belongL = node->keynum;
        int belongR = node->keynum;
        for(int i = 0 ; i < node->keynum; i++){
            if(rangeL < valueList_[node->key[i]] ||
               (rangeL == valueList_[node->key[i]] && (rangeL == valueList_[findRight(node->child[i])]))){
                belongL = i;
                break;
            }
        }
        for(int i = 0 ; i < node->keynum; i++){
            if(rangeR < valueList_[node->key[i]]){
                belongR = i;
                break;
            }
        }
        if(belongL == belongR) return findHighNode(node->child[belongL], rangeL, rangeR);
        return node;
    }


    void getNeighborsByHeuristic2(
            ResultHeap &top_candidates,
            const size_t M) {
        if (top_candidates.size() < M) {
            return;
        }

        std::priority_queue<std::pair<float, tableint>> queue_closest;
        std::vector<std::pair<float, tableint>> return_list;
        while (top_candidates.size() > 0) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (queue_closest.size()) {
            if (return_list.size() >= M)
                break;
            std::pair<float, tableint> curent_pair = queue_closest.top();
            float dist_to_query = -curent_pair.first;
            queue_closest.pop();
            bool good = true;

            for (std::pair<float, tableint> second_pair : return_list) {
                float curdist =
                        fstdistfunc_(getDataByInternalId(second_pair.second),
                                     getDataByInternalId(curent_pair.second),
                                     dist_func_param_);
                if (curdist < dist_to_query) {
                    good = false;
                    break;
                }
            }
            if (good) {
                return_list.push_back(curent_pair);
            }
        }

        for (std::pair<float, tableint> curent_pair : return_list) {
            top_candidates.emplace(-curent_pair.first, curent_pair.second);
        }
    }

    inline char *getDataByInternalId(tableint internal_id) const {
        return (char*)(vecData_ + internal_id * data_size_);
    }

    ResultHeap searchBaseLayer(const std::vector<tableint> &ep_ids, const void *data_point, int layer) {
        tag ++;

        ResultHeap top_candidates;
        ResultHeap candidateSet;

        float lowerBound;

        for(int i = 0; i < ep_ids.size(); i++) {
            int ep_id = ep_ids[i];
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            if(!isDeleted[ep_id]) {
                top_candidates.emplace(dist, ep_id);
                candidateSet.emplace(-dist, ep_id);
            }
            else{
                candidateSet.emplace(-std::numeric_limits<float>::max(), ep_id);
            }
            visited_array[ep_id] = tag;
        }

        if(!top_candidates.empty())
            lowerBound = top_candidates.top().first;
        else
            lowerBound = std::numeric_limits<float>::max();


        while (!candidateSet.empty()) {
            std::pair<float, tableint> curr_el_pair = candidateSet.top();
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == ef_construction) {
                break;
            }
            candidateSet.pop();

            tableint curNodeNum = curr_el_pair.second;

            for(int i = 0; i <= 0; i++) {
                if(layer - i <= 0) break;
                int *data = (int *) get_linklist(curNodeNum, layer - i);
                size_t size = getListCount((linklistsizeint *) data);
                tableint *datal = (tableint *) (data + 1);

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + *(datal + j + 2)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 2)), _MM_HINT_T0);
                    // _mm_prefetch((char *) (visited_array + *(datal + j + 3)), _MM_HINT_T0);
                    // _mm_prefetch(getDataByInternalId(*(datal + j + 3)), _MM_HINT_T0);
                    // _mm_prefetch((char *) (visited_array + *(datal + j + 4)), _MM_HINT_T0);
                    // _mm_prefetch(getDataByInternalId(*(datal + j + 4)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == tag) continue;
                    visited_array[candidate_id] = tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if (top_candidates.size() < ef_construction || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, candidate_id);
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if(!isDeleted[candidate_id])
                            top_candidates.emplace(dist1, candidate_id);
                        if (top_candidates.size() > ef_construction)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;
                    }
                }
            }
        }

        return top_candidates;
    }

    ResultHeap
    searchBaseLayer0(std::vector<tableint> ep_ids, const void *data_point, int Layer, int rangeL, int rangeR, int ef, int splitPoint) {
        tag ++;

        ResultHeap top_candidates;
        ResultHeap candidateSet;

        float lowerBound;
        for(int i = 0; i < (int)ep_ids.size(); i++) {
            int ep_id = ep_ids[i];
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            if(!isDeleted[ep_id] && valueList_[ep_id]>=rangeL && valueList_[ep_id] <= rangeR) {
                top_candidates.emplace(dist, ep_id);
                candidateSet.emplace(-dist, ep_id);
            }
            else{
                candidateSet.emplace(-std::numeric_limits<float>::max(), ep_id);
            }
            visited_array[ep_id] = tag;

            // Expand cluster members for entry points at layers >= 1
            if(searchLayer[ep_id] >= 1 && searchLayer[ep_id] <= maxLayer){
                expandClusterMembers(ep_id, searchLayer[ep_id], data_point, rangeL, rangeR, ef,
                                     candidateSet, top_candidates, lowerBound);
            }
        }

        if(!top_candidates.empty())
            lowerBound = top_candidates.top().first;
        else
            lowerBound = std::numeric_limits<float>::max();

        while (!candidateSet.empty()) {
            std::pair<float, tableint> curr_el_pair = candidateSet.top();
            tableint curNodeNum = curr_el_pair.second;
            short int layer = searchLayer[curNodeNum];
            if ((-curr_el_pair.first) > lowerBound && (int)top_candidates.size() == ef) {
                break;
            }
            candidateSet.pop();

            for(int i = 0; i <= 1; i++) {
                if(layer - i <= 0) break;
                int curLayer = layer - i;
                int *data = (int *) get_linklist(curNodeNum, curLayer);

                size_t size = getListCount((linklistsizeint *) data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                        _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                        _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
                        _mm_prefetch((char *) (visited_array + *(datal + j + 2)), _MM_HINT_T0);
                        _mm_prefetch(getDataByInternalId(*(datal + j + 2)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == tag) continue;
                    visited_array[candidate_id] = tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    tableint cid = candidate_id;

                    float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if ((int)top_candidates.size() < ef || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, cid);
                        searchLayer[cid] = layer;
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if(!isDeleted[candidate_id])
                            if ( valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR)
                                top_candidates.emplace(dist1, cid);

                        if ((int)top_candidates.size() > ef)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;

                        // Cluster expansion: when we reach a point via a layer > 0 edge,
                        // also expand all members of that point's cluster
                        if(curLayer >= 1 && curLayer <= maxLayer){
                            expandClusterMembers(cid, curLayer, data_point, rangeL, rangeR, ef,
                                                 candidateSet, top_candidates, lowerBound);
                        }
                    }
                }
            }

            if(splitPoint!=-1) {
                int *data = (int *) get_linklist(curNodeNum, Layer);

                size_t size = getListCount((linklistsizeint *) data);
                tableint *datal = (tableint *) (data + 1);
#ifdef USE_SSE
                _mm_prefetch((char *) (visited_array + *(data + 1)), _MM_HINT_T0);
                _mm_prefetch((char *) (visited_array + *(data + 1) + 64), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*datal), _MM_HINT_T0);
                _mm_prefetch(getDataByInternalId(*(datal + 1)), _MM_HINT_T0);
#endif

                for (size_t j = 0; j < size; j++) {
                    tableint candidate_id = *(datal + j);
#ifdef USE_SSE
                    _mm_prefetch((char *) (visited_array + *(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 1)), _MM_HINT_T0);
                    _mm_prefetch((char *) (visited_array + *(datal + j + 2)), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(*(datal + j + 2)), _MM_HINT_T0);
#endif
                    if (visited_array[candidate_id] == tag) continue;
                    if ( !(valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR))continue;
                    visited_array[candidate_id] = tag;
                    char *currObj1 = (getDataByInternalId(candidate_id));

                    tableint cid = candidate_id;

                    float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                    if ((int)top_candidates.size() < ef || lowerBound > dist1) {
                        candidateSet.emplace(-dist1, cid);
                        searchLayer[cid] = searchLayer[ep_ids[0]] == layer ? searchLayer[ep_ids[1]]: searchLayer[ep_ids[0]];
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif

                        if ( valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR)
                            top_candidates.emplace(dist1, cid);

                        if ((int)top_candidates.size() > ef)
                            top_candidates.pop();

                        if (!top_candidates.empty())
                            lowerBound = top_candidates.top().first;

                        // Cluster expansion for LCA layer edges too
                        if(Layer >= 1 && Layer <= maxLayer){
                            expandClusterMembers(cid, Layer, data_point, rangeL, rangeR, ef,
                                                 candidateSet, top_candidates, lowerBound);
                        }
                    }
                }
            }
        }

        return top_candidates;
    }

    // Expand all members of the cluster containing pointId at the given layer
    void expandClusterMembers(tableint pointId, int layer, const void* data_point,
                              int rangeL, int rangeR, int ef,
                              ResultHeap& candidateSet, ResultHeap& top_candidates,
                              float& lowerBound) {
        if(layer < 1 || layer > maxLayer) return;
        if(layer - 1 >= (int)clusterLayers.size()) return;
        const ClusterLayer& cl = clusterLayers[layer - 1];
        if(pointId >= (int)cl.pointToCluster.size()) return;
        int clId = cl.pointToCluster[pointId];
        if(clId < 0 || clId >= (int)cl.members.size()) return;

        const std::vector<tableint>& members = cl.members[clId];
        for(tableint member : members){
            if(visited_array[member] == tag) continue;
            visited_array[member] = tag;

            float dist1 = fstdistfunc_(data_point, getDataByInternalId(member), dist_func_param_);
            if((int)top_candidates.size() < ef || lowerBound > dist1){
                candidateSet.emplace(-dist1, member);
                searchLayer[member] = (short int)layer;

                if(!isDeleted[member])
                    if(valueList_[member] >= rangeL && valueList_[member] <= rangeR)
                        top_candidates.emplace(dist1, member);

                if((int)top_candidates.size() > ef)
                    top_candidates.pop();

                if(!top_candidates.empty())
                    lowerBound = top_candidates.top().first;
            }
        }
    }

    tableint
    findEntry(const void *query_data, node *nd, tableint currObj) const {
        float curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        int endLayer = nd->layer;
        int startLayer = findEntryLayer(endLayer);

        for (int layer = startLayer; layer < endLayer; layer += skipLayer) {
            bool changed = true;
            while (changed) {
                changed = false;
                unsigned int *data;

                for(int l = 0; l <= 0 ; l++){
                    data = (unsigned int *) get_linklist(currObj, layer-l);
                    int size = getListCount(data);

                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
                        _mm_prefetch(getDataByInternalId(*(datal + i + 2)), _MM_HINT_T0);
#endif
                        float d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);

                        if (d < curdist) {
                            curdist = d;
                            currObj = cand;
                            changed = true;
                        }
                    }
                }
            }
        }
        return currObj;
    }


    linklistsizeint *get_linklist(tableint internal_id, int layer) const {
        if(layer == 0){
            return (linklistsizeint *) linklist[internal_id];
        }
        // Layer >= 1: route through cluster edge list
        if(layer - 1 < (int)clusterLayers.size()){
            const ClusterLayer& cl = clusterLayers[layer - 1];
            if(internal_id < (int)cl.pointToCluster.size()){
                int clId = cl.pointToCluster[internal_id];
                if(clId >= 0 && clId < (int)cl.edgeList.size()){
                    return (linklistsizeint *) cl.edgeList[clId];
                }
            }
        }
        // Fallback: return layer-0 edges
        return (linklistsizeint *) linklist[internal_id];
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }

    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }
};


#endif //RANGEHNSW_RANGEHNSW_HPP
