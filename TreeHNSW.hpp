#ifndef RANGEHNSW_RANGEHNSW_HPP
#define RANGEHNSW_RANGEHNSW_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <numeric>
#include <cmath>
#include <limits>
#include <queue>

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

        // Only allocate layer-0 per-point storage (O(n) total instead of O(n*logn))
        sizeLinkList = (M * sizeof(tableint) + sizeof(linklistsizeint));

        space = hnswlib::L2Space(dim);

        sortedArray.reserve(eleNum);

        for(int i = 0; i < (int)eleNum; i++){
            key2Id[keyList[i]] = i;
            linklist[i] = (char *) malloc(sizeLinkList);
            sortedArray.push_back(i);
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
            top.push({r.first,keyList_[r.second]});
        }
        return top;
    }

    void addPoint(int key,int value, char* data){
        keyList_[eleCount] = key;
        key2Id[key] = eleCount;
        valueList_[eleCount] = value;
        memcpy(vecData_+ dim * sizeof(float) * eleCount, data, dim * sizeof(float));
        linklist[eleCount] = (char *) malloc(sizeLinkList);
        unsigned int *newListData = (unsigned int *) get_linklist(eleCount);
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
        for(int i = 0; i < (int)maxNum; i++){
            linklist[i] = (char *) realloc(linklist[i], sizeLinkList);
        }
        for(int i = maxNum; i < maxEleNum; i++){
            linklist[i] = (char *) malloc(sizeLinkList);
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
        short int layer;
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

    struct ClusterInfo {
        std::vector<float> centroids;               // numClusters * dim floats
        std::vector<std::vector<tableint>> members; // members[k] = point IDs in cluster k
        std::vector<tableint> point2cluster;        // point2cluster[pointId] = cluster ID
        int numClusters = 0;
    };

    // layerClusters[layer] for layer >= 1
    std::vector<ClusterInfo> layerClusters;
    // clusterLinklist[layer][clusterId] = list of neighbor cluster IDs
    std::vector<std::vector<std::vector<tableint>>> clusterLinklist;

    // -------------------------------------------------------
    // Helper: squared L2 distance between two float vectors
    // -------------------------------------------------------
    inline float l2DistCentroid(const float* a, const float* b) const {
        float s = 0.0f;
        for (int i = 0; i < dim; i++) {
            float diff = a[i] - b[i];
            s += diff * diff;
        }
        return s;
    }

    // -------------------------------------------------------
    // KMeans clustering using L2 distance
    // clusterSize ~= log(N), so numClusters ~= N / log(N)
    // -------------------------------------------------------
    ClusterInfo kmeansCluster(const std::vector<tableint>& pointIds, int clusterSize, int maxIter = 20) {
        int n = (int)pointIds.size();
        int K = std::max(1, n / std::max(1, clusterSize));

        ClusterInfo info;
        info.numClusters = K;
        info.centroids.resize((size_t)K * dim, 0.0f);
        info.members.resize(K);
        info.point2cluster.resize(maxNum, 0);

        if (n == 0 || K == 0) return info;

        // Initialize centroids: pick K distinct random point vectors.
        // K = n / clusterSize <= n, so the shuffle gives K distinct indices.
        std::vector<int> idxs(n);
        std::iota(idxs.begin(), idxs.end(), 0);
        std::shuffle(idxs.begin(), idxs.end(), eng);
        for (int k = 0; k < K; k++) {
            const float* src = (const float*)getDataByInternalId(pointIds[idxs[k]]);
            std::copy(src, src + dim, info.centroids.data() + (size_t)k * dim);
        }

        // KMeans iterations
        for (int iter = 0; iter < maxIter; iter++) {
            for (int k = 0; k < K; k++) info.members[k].clear();

            for (int i = 0; i < n; i++) {
                tableint pid = pointIds[i];
                const float* pvec = (const float*)getDataByInternalId(pid);
                float bestDist = std::numeric_limits<float>::max();
                int bestK = 0;
                for (int k = 0; k < K; k++) {
                    float d = l2DistCentroid(pvec, info.centroids.data() + (size_t)k * dim);
                    if (d < bestDist) { bestDist = d; bestK = k; }
                }
                info.point2cluster[pid] = (tableint)bestK;
                info.members[bestK].push_back(pid);
            }

            for (int k = 0; k < K; k++) {
                if (info.members[k].empty()) {
                    // Re-seed empty cluster from a random point
                    std::uniform_int_distribution<> distr(0, n - 1);
                    const float* src = (const float*)getDataByInternalId(pointIds[distr(eng)]);
                    std::copy(src, src + dim, info.centroids.data() + (size_t)k * dim);
                    continue;
                }
                float* c = info.centroids.data() + (size_t)k * dim;
                std::fill(c, c + dim, 0.0f);
                for (tableint pid : info.members[k]) {
                    const float* v = (const float*)getDataByInternalId(pid);
                    for (int d = 0; d < dim; d++) c[d] += v[d];
                }
                float inv = 1.0f / (float)info.members[k].size();
                for (int d = 0; d < dim; d++) c[d] *= inv;
            }
        }

        return info;
    }

    // -------------------------------------------------------
    // Heuristic neighbor selection using centroid distances
    // Mirrors getNeighborsByHeuristic2 but operates on cluster centroids
    // -------------------------------------------------------
    void getNeighborsByHeuristic2Cluster(
            ResultHeap& top_candidates,
            const size_t Mmax,
            const std::vector<float>& centroids) {
        if (top_candidates.size() < Mmax) return;

        std::priority_queue<std::pair<float, tableint>> queue_closest;
        std::vector<std::pair<float, tableint>> return_list;
        while (!top_candidates.empty()) {
            queue_closest.emplace(-top_candidates.top().first, top_candidates.top().second);
            top_candidates.pop();
        }

        while (!queue_closest.empty()) {
            if (return_list.size() >= Mmax) break;
            auto cur = queue_closest.top();
            float dist_to_query = -cur.first;
            queue_closest.pop();
            bool good = true;
            for (auto& sec : return_list) {
                float curdist = l2DistCentroid(
                        centroids.data() + (size_t)sec.second * dim,
                        centroids.data() + (size_t)cur.second * dim);
                if (curdist < dist_to_query) { good = false; break; }
            }
            if (good) return_list.push_back(cur);
        }

        for (auto& p : return_list) {
            top_candidates.emplace(-p.first, p.second);
        }
    }

    // -------------------------------------------------------
    // Build HNSW graph between cluster centroids at a given layer.
    // Uses getNeighborsByHeuristic2Cluster for M-neighbor selection.
    // -------------------------------------------------------
    void buildClusterGraph(int layer) {
        ClusterInfo& info = layerClusters[layer];
        int K = info.numClusters;
        clusterLinklist[layer].assign(K, std::vector<tableint>());

        if (K <= 1) return;

        std::vector<unsigned int> clusterVisited(K, 0);
        unsigned int cvTag = 0;

        // Insert clusters one by one into the growing graph
        for (int cid = 1; cid < K; cid++) {
            cvTag++;
            const float* qvec = info.centroids.data() + (size_t)cid * dim;

            // Greedy search to find a good entry cluster among those already inserted
            int entryCid = 0;
            float entryDist = l2DistCentroid(qvec, info.centroids.data());
            {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (tableint nb : clusterLinklist[layer][entryCid]) {
                        float d = l2DistCentroid(qvec, info.centroids.data() + (size_t)nb * dim);
                        if (d < entryDist) { entryDist = d; entryCid = (int)nb; changed = true; }
                    }
                }
            }

            // Beam search to find ef_construction nearest neighbours
            ResultHeap top_candidates;
            ResultHeap candidateSet;
            clusterVisited[entryCid] = cvTag;
            top_candidates.emplace(entryDist, (tableint)entryCid);
            candidateSet.emplace(-entryDist, (tableint)entryCid);
            float lowerBound = entryDist;

            while (!candidateSet.empty()) {
                auto curr = candidateSet.top();
                if (-curr.first > lowerBound && (int)top_candidates.size() >= ef_construction) break;
                candidateSet.pop();
                int cur = (int)curr.second;

                for (tableint nb : clusterLinklist[layer][cur]) {
                    int inb = (int)nb;
                    if (clusterVisited[inb] == cvTag) continue;
                    clusterVisited[inb] = cvTag;
#ifdef USE_SSE
                    _mm_prefetch((char*)(info.centroids.data() + (size_t)inb * dim), _MM_HINT_T0);
#endif
                    float d = l2DistCentroid(qvec, info.centroids.data() + (size_t)inb * dim);
                    if ((int)top_candidates.size() < ef_construction || d < lowerBound) {
                        candidateSet.emplace(-d, nb);
                        top_candidates.emplace(d, nb);
                        if ((int)top_candidates.size() > ef_construction) top_candidates.pop();
                        if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                    }
                }
            }

            // Select M neighbours using the heuristic
            getNeighborsByHeuristic2Cluster(top_candidates, M, info.centroids);

            // Add bidirectional edges
            while (!top_candidates.empty()) {
                int nb = (int)top_candidates.top().second;
                top_candidates.pop();
                clusterLinklist[layer][cid].push_back((tableint)nb);
                if ((int)clusterLinklist[layer][nb].size() < M) {
                    clusterLinklist[layer][nb].push_back((tableint)cid);
                }
            }
        }
    }

    // -------------------------------------------------------
    // Search the cluster graph: navigates cluster centroids then
    // expands members. Used by searchBaseLayer for layers >= 1.
    // -------------------------------------------------------
    ResultHeap searchClusterLayer(const std::vector<tableint>& ep_ids,
                                   const void* data_point, int layer) {
        tag++;
        const ClusterInfo& clInfo = layerClusters[layer];
        int K = clInfo.numClusters;

        ResultHeap top_candidates;
        if (K == 0) return top_candidates;

        using CluPair = std::pair<float, int>;
        std::priority_queue<CluPair, std::vector<CluPair>, std::greater<CluPair>> clusterQueue;
        std::vector<unsigned int> clusterVisited(K, 0);
        unsigned int cvTag = 1;

        float lowerBound = std::numeric_limits<float>::max();

        for (tableint ep_id : ep_ids) {
            if (ep_id >= (tableint)clInfo.point2cluster.size()) continue;
            int cid = (int)clInfo.point2cluster[ep_id];
            if (cid >= K || clusterVisited[cid] == cvTag) continue;
            clusterVisited[cid] = cvTag;
            float d = l2DistCentroid((const float*)data_point,
                                     clInfo.centroids.data() + (size_t)cid * dim);
            clusterQueue.push({d, cid});
        }

        while (!clusterQueue.empty()) {
            auto [centDist, cid] = clusterQueue.top();
            clusterQueue.pop();

            if (centDist > lowerBound && (int)top_candidates.size() >= ef_construction) break;

            const auto& members = clInfo.members[cid];
            for (size_t j = 0; j < members.size(); j++) {
                tableint mid = members[j];
#ifdef USE_SSE
                if (j + 1 < members.size()) {
                    _mm_prefetch((char*)(visited_array + members[j + 1]), _MM_HINT_T0);
                    _mm_prefetch(getDataByInternalId(members[j + 1]), _MM_HINT_T0);
                }
#endif
                if (visited_array[mid] == tag) continue;
                visited_array[mid] = tag;

                float dist = fstdistfunc_(data_point, getDataByInternalId(mid), dist_func_param_);
                if (!isDeleted[mid]) {
                    if ((int)top_candidates.size() < ef_construction || dist < lowerBound) {
                        top_candidates.emplace(dist, mid);
                        if ((int)top_candidates.size() > ef_construction) top_candidates.pop();
                        if (!top_candidates.empty()) lowerBound = top_candidates.top().first;
                    }
                }
            }

            for (tableint nb : clusterLinklist[layer][cid]) {
                int inb = (int)nb;
                if (inb >= K || clusterVisited[inb] == cvTag) continue;
                clusterVisited[inb] = cvTag;
#ifdef USE_SSE
                _mm_prefetch((char*)(clInfo.centroids.data() + (size_t)inb * dim), _MM_HINT_T0);
#endif
                float nd = l2DistCentroid((const float*)data_point,
                                          clInfo.centroids.data() + (size_t)inb * dim);
                if ((int)top_candidates.size() < ef_construction || nd < lowerBound) {
                    clusterQueue.push({nd, inb});
                }
            }
        }

        return top_candidates;
    }

    // -------------------------------------------------------
    // Add a dynamically inserted point to cluster memberships
    // -------------------------------------------------------
    void addPointToClusters(tableint id) {
        for (int layer = 1; layer <= maxLayer; layer++) {
            if (layer >= (int)layerClusters.size()) break;
            ClusterInfo& clInfo = layerClusters[layer];
            if (clInfo.numClusters == 0) continue;

            const float* pvec = (const float*)getDataByInternalId(id);

            // Greedy search on cluster graph to find nearest cluster
            int curC = 0;
            float curDist = l2DistCentroid(pvec, clInfo.centroids.data());
            {
                bool changed = true;
                while (changed) {
                    changed = false;
                    for (tableint nb : clusterLinklist[layer][curC]) {
                        float d = l2DistCentroid(pvec, clInfo.centroids.data() + (size_t)nb * dim);
                        if (d < curDist) { curDist = d; curC = (int)nb; changed = true; }
                    }
                }
            }

            // Extend point2cluster if needed
            if (id >= (tableint)clInfo.point2cluster.size()) {
                clInfo.point2cluster.resize(id + 1, 0);
            }
            clInfo.point2cluster[id] = (tableint)curC;
            clInfo.members[curC].push_back(id);

            // Update centroid (running average)
            int sz = (int)clInfo.members[curC].size();
            float* c = clInfo.centroids.data() + (size_t)curC * dim;
            for (int d = 0; d < dim; d++) {
                c[d] = (c[d] * (sz - 1) + pvec[d]) / (float)sz;
            }
        }
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
            unsigned int *newListData = (unsigned int *) get_linklist(sortedArray[i]);
            setListCount(newListData, 0);
        }

        // Pre-allocate cluster structures
        layerClusters.resize(maxLayer + 1);
        clusterLinklist.resize(maxLayer + 1);

        // clusterSize ~= log(N) so numClusters ~= N/log(N)
        int clusterSize = std::max(2, (int)std::round(std::log((float)std::max(2, eleNum))));

        while(q[qid].size() > 1){
            std::cout<<"layer:"<<q[qid].front().second->layer<<std::endl;
            int nxtqid = qid ^ 1;

            int layer = q[qid].front().second->layer + 1;

            // Build KMeans cluster graph for this layer
            {
                std::vector<tableint> layerPoints;
                layerPoints.reserve(eleNum);
                // Copy queue to collect all point IDs without consuming it
                std::queue<std::pair<std::pair<int,int>, node*>> tmpq = q[qid];
                while (!tmpq.empty()) {
                    auto& item = tmpq.front();
                    for (int ii = item.first.first; ii <= item.first.second; ii++) {
                        layerPoints.push_back(sortedArray[ii]);
                    }
                    tmpq.pop();
                }

                layerClusters[layer] = kmeansCluster(layerPoints, clusterSize);
                buildClusterGraph(layer);

                // Track edges: each cluster has up to M edges
                numEdges += layerClusters[layer].numClusters * M;
            }

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
                nd->keynum = numChild - 1;
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
                nd->layer = nd->child[0]->layer + 1;

                // No per-point edge building at layers >= 1: cluster graph is used instead

                q[nxtqid].push({{tmp[0].first,tmp[tmp.size() - 1].second}, nd});
            }

            qid = nxtqid;
        }
        std::cout<<"cluster graph built, total cluster edges:"<<numEdges<<std::endl;
        std::cout<<"average cluster edges per point:"<<numEdges*1.0/eleNum<<std::endl;
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
                // Build cluster structure for the new root layer
                int newLayer = newRoot->layer;
                if (newLayer >= (int)layerClusters.size()) {
                    layerClusters.resize(newLayer + 1);
                    clusterLinklist.resize(newLayer + 1);
                    std::vector<tableint> allPoints;
                    allPoints.reserve(eleCount);
                    for (int i = 0; i < (int)eleCount; i++) allPoints.push_back((tableint)i);
                    int cs = std::max(2, (int)std::round(std::log((float)std::max(2, (int)eleCount))));
                    layerClusters[newLayer] = kmeansCluster(allPoints, cs);
                    buildClusterGraph(newLayer);
                }
                splitNode(newRoot, 0);
            }
        }
        // Update cluster memberships for this new point at every layer >= 1
        addPointToClusters((tableint)id);
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

    // Simplified refresh: update B-tree entry point only.
    // Cluster graph edges are fixed; only membership changes via addPointToClusters.
    void refresh(node *nd, int){
        updateEntry(nd);
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
            // Insert into B-tree structure; no per-point edges for layers >= 1
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

    // Simplified refresh: update entry point only
    void refresh(node *nd){
        updateEntry(nd);
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

    // -------------------------------------------------------
    // searchBaseLayer: used during construction and dynamic ops.
    // For layers >= 1: delegates to searchClusterLayer.
    // For layer 0: layer 0 has no edges; returns ep_ids as candidates.
    // -------------------------------------------------------
    ResultHeap searchBaseLayer(const std::vector<tableint> &ep_ids, const void *data_point, int layer) {
        if (layer >= 1 && layer < (int)layerClusters.size() && layerClusters[layer].numClusters > 0) {
            return searchClusterLayer(ep_ids, data_point, layer);
        }

        // Layer 0: no edges; evaluate entry points directly
        tag++;
        ResultHeap top_candidates;
        for (tableint ep_id : ep_ids) {
            if (visited_array[ep_id] == tag) continue;
            visited_array[ep_id] = tag;
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            if (!isDeleted[ep_id]) {
                top_candidates.emplace(dist, ep_id);
            }
        }
        return top_candidates;
    }

    // -------------------------------------------------------
    // searchBaseLayer0: main range-query search with LCA support.
    // Preserves:
    //   - searchLayer tracking for cross-layer LCA navigation
    //   - splitPoint cross-subtree expansion
    //   - all _mm_prefetch instructions
    //   - range filter (valueList_)
    // Replaces per-point linklist traversal with cluster graph traversal.
    // -------------------------------------------------------
    ResultHeap
    searchBaseLayer0(std::vector<tableint> ep_ids, const void *data_point, int Layer, int rangeL, int rangeR, int ef, int splitPoint) {
        tag ++;

        ResultHeap top_candidates;
        ResultHeap candidateSet;

        float lowerBound;
        for(tableint ep_id : ep_ids) {
            float dist = fstdistfunc_(data_point, getDataByInternalId(ep_id), dist_func_param_);
            if(!isDeleted[ep_id] && valueList_[ep_id]>=rangeL && valueList_[ep_id] <= rangeR) {
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
            tableint curNodeNum = curr_el_pair.second;
            short int layer = searchLayer[curNodeNum];
            if ((-curr_el_pair.first) > lowerBound && top_candidates.size() == (size_t)ef) {
                break;
            }
            candidateSet.pop();

            // -------------------------------------------------------
            // Main neighbour expansion via cluster graph (two layers)
            // -------------------------------------------------------
            for(int i = 0; i <= 1; i++) {
                int curLayer = layer - i;
                if(curLayer <= 0) break;

                if (curLayer < (int)layerClusters.size() && layerClusters[curLayer].numClusters > 0) {
                    const ClusterInfo& clInfo = layerClusters[curLayer];
                    int clusterId = (curNodeNum < (tableint)clInfo.point2cluster.size())
                                        ? (int)clInfo.point2cluster[curNodeNum] : 0;
#ifdef USE_SSE
                    if (!clusterLinklist[curLayer][clusterId].empty()) {
                        int fNb = (int)clusterLinklist[curLayer][clusterId][0];
                        if (!clInfo.members[fNb].empty()) {
                            _mm_prefetch((char*)(visited_array + clInfo.members[fNb][0]), _MM_HINT_T0);
                            _mm_prefetch(getDataByInternalId(clInfo.members[fNb][0]), _MM_HINT_T0);
                        }
                    }
#endif
                    for (tableint nbCluster : clusterLinklist[curLayer][clusterId]) {
                        const auto& members = clInfo.members[(int)nbCluster];
#ifdef USE_SSE
                        if (!members.empty()) {
                            _mm_prefetch((char*)(visited_array + members[0]), _MM_HINT_T0);
                            _mm_prefetch(getDataByInternalId(members[0]), _MM_HINT_T0);
                        }
#endif
                        for (size_t j = 0; j < members.size(); j++) {
                            tableint candidate_id = members[j];
#ifdef USE_SSE
                            if (j + 1 < members.size()) {
                                _mm_prefetch((char*)(visited_array + members[j + 1]), _MM_HINT_T0);
                                _mm_prefetch(getDataByInternalId(members[j + 1]), _MM_HINT_T0);
                            }
#endif
                            if (visited_array[candidate_id] == tag) continue;
                            visited_array[candidate_id] = tag;
                            char *currObj1 = getDataByInternalId(candidate_id);

                            float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                            if (top_candidates.size() < (size_t)ef || lowerBound > dist1) {
                                candidateSet.emplace(-dist1, candidate_id);
                                searchLayer[candidate_id] = (short int)curLayer;
#ifdef USE_SSE
                                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif
                                if(!isDeleted[candidate_id])
                                    if (valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR)
                                        top_candidates.emplace(dist1, candidate_id);

                                if (top_candidates.size() > (size_t)ef)
                                    top_candidates.pop();

                                if (!top_candidates.empty())
                                    lowerBound = top_candidates.top().first;
                            }
                        }
                    }
                }
                // layer 0 has no edges; no fallback needed
            }

            // -------------------------------------------------------
            // LCA split-point cross-subtree expansion (preserved)
            // -------------------------------------------------------
            if(splitPoint != -1) {
                int clLayer = Layer;
                if (clLayer >= 1 && clLayer < (int)layerClusters.size() &&
                    layerClusters[clLayer].numClusters > 0) {
                    const ClusterInfo& clInfo = layerClusters[clLayer];
                    int clusterId = (curNodeNum < (tableint)clInfo.point2cluster.size())
                                        ? (int)clInfo.point2cluster[curNodeNum] : 0;
#ifdef USE_SSE
                    if (!clusterLinklist[clLayer][clusterId].empty()) {
                        int fNb = (int)clusterLinklist[clLayer][clusterId][0];
                        if (!clInfo.members[fNb].empty()) {
                            _mm_prefetch((char*)(visited_array + clInfo.members[fNb][0]), _MM_HINT_T0);
                            _mm_prefetch(getDataByInternalId(clInfo.members[fNb][0]), _MM_HINT_T0);
                        }
                    }
#endif
                    for (tableint nbCluster : clusterLinklist[clLayer][clusterId]) {
                        const auto& members = clInfo.members[(int)nbCluster];
#ifdef USE_SSE
                        if (!members.empty()) {
                            _mm_prefetch((char*)(visited_array + members[0]), _MM_HINT_T0);
                            _mm_prefetch(getDataByInternalId(members[0]), _MM_HINT_T0);
                        }
#endif
                        for (size_t j = 0; j < members.size(); j++) {
                            tableint candidate_id = members[j];
#ifdef USE_SSE
                            if (j + 1 < members.size()) {
                                _mm_prefetch((char*)(visited_array + members[j + 1]), _MM_HINT_T0);
                                _mm_prefetch(getDataByInternalId(members[j + 1]), _MM_HINT_T0);
                            }
#endif
                            if (visited_array[candidate_id] == tag) continue;
                            if (!(valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR)) continue;
                            visited_array[candidate_id] = tag;
                            char *currObj1 = getDataByInternalId(candidate_id);

                            float dist1 = fstdistfunc_(data_point, currObj1, dist_func_param_);
                            if (top_candidates.size() < (size_t)ef || lowerBound > dist1) {
                                candidateSet.emplace(-dist1, candidate_id);
                                // Assign the "other side" layer (LCA cross-subtree logic preserved)
                                searchLayer[candidate_id] = (ep_ids.size() >= 2 && searchLayer[ep_ids[0]] == layer)
                                        ? searchLayer[ep_ids[1]] : searchLayer[ep_ids[0]];
#ifdef USE_SSE
                                _mm_prefetch(getDataByInternalId(candidateSet.top().second), _MM_HINT_T0);
#endif
                                if (valueList_[candidate_id] >= rangeL && valueList_[candidate_id] <= rangeR)
                                    top_candidates.emplace(dist1, candidate_id);

                                if (top_candidates.size() > (size_t)ef)
                                    top_candidates.pop();

                                if (!top_candidates.empty())
                                    lowerBound = top_candidates.top().first;
                            }
                        }
                    }
                }
            }
        }

        return top_candidates;
    }

    // -------------------------------------------------------
    // findEntry: greedy descent to find best entry point in nd's subtree.
    // For layers >= 1: navigate the cluster graph, then expand members.
    // Preserves _mm_prefetch instructions.
    // -------------------------------------------------------
    tableint
    findEntry(const void *query_data, node *nd, tableint currObj) const {
        float curdist = fstdistfunc_(query_data, getDataByInternalId(currObj), dist_func_param_);
        int endLayer = nd->layer;
        int startLayer = findEntryLayer(endLayer);

        for (int layer = startLayer; layer < endLayer; layer += skipLayer) {

            if (layer >= 1 && layer < (int)layerClusters.size() &&
                layerClusters[layer].numClusters > 0) {
                // -------------------------------------------------------
                // Cluster-graph greedy descent with prefetch
                // -------------------------------------------------------
                const ClusterInfo& clInfo = layerClusters[layer];
                int curCluster = (currObj < (tableint)clInfo.point2cluster.size())
                                     ? (int)clInfo.point2cluster[currObj] : 0;

                // Phase 1: navigate cluster graph to find nearest cluster centroid
                float centDist = l2DistCentroid(
                        (const float*)query_data,
                        clInfo.centroids.data() + (size_t)curCluster * dim);
                {
                    bool changed = true;
                    while (changed) {
                        changed = false;
                        for (tableint nb : clusterLinklist[layer][curCluster]) {
                            int inb = (int)nb;
#ifdef USE_SSE
                            _mm_prefetch((char*)(clInfo.centroids.data() + (size_t)inb * dim), _MM_HINT_T0);
#endif
                            float d = l2DistCentroid(
                                    (const float*)query_data,
                                    clInfo.centroids.data() + (size_t)inb * dim);
                            if (d < centDist) { centDist = d; curCluster = inb; changed = true; }
                        }
                    }
                }

                // Phase 2: expand best cluster to find nearest actual point
                const auto& bestMembers = clInfo.members[curCluster];
                for (size_t j = 0; j < bestMembers.size(); j++) {
                    tableint member = bestMembers[j];
#ifdef USE_SSE
                    if (j + 1 < bestMembers.size())
                        _mm_prefetch(getDataByInternalId(bestMembers[j + 1]), _MM_HINT_T0);
#endif
                    float d = fstdistfunc_(query_data, getDataByInternalId(member), dist_func_param_);
                    if (d < curdist) { curdist = d; currObj = member; }
                }

            } else {
                // Fallback (layer 0 or uninitialized): per-point navigation
                bool changed = true;
                while (changed) {
                    changed = false;
                    unsigned int *data = (unsigned int *) get_linklist(currObj);
                    int size = getListCount(data);
                    tableint *datal = (tableint *) (data + 1);
                    for (int i = 0; i < size; i++) {
                        tableint cand = datal[i];
#ifdef USE_SSE
                        _mm_prefetch(getDataByInternalId(*(datal + i + 1)), _MM_HINT_T0);
                        _mm_prefetch(getDataByInternalId(*(datal + i + 2)), _MM_HINT_T0);
#endif
                        float d = fstdistfunc_(query_data, getDataByInternalId(cand), dist_func_param_);
                        if (d < curdist) { curdist = d; currObj = cand; changed = true; }
                    }
                }
            }
        }
        return currObj;
    }


    // All per-point storage lives in a single sizeLinkList block (layer 0 only).
    linklistsizeint *get_linklist(tableint internal_id) const {
        return (linklistsizeint *) (linklist[internal_id]);
    }


    unsigned short int getListCount(linklistsizeint * ptr) const {
        return *((unsigned short int *)ptr);
    }

    void setListCount(linklistsizeint * ptr, unsigned short int size) const {
        *((unsigned short int*)(ptr))=*((unsigned short int *)&size);
    }
};


#endif //RANGEHNSW_RANGEHNSW_HPP
