#ifndef RANGEHNSW_RANGEHNSW_HPP
#define RANGEHNSW_RANGEHNSW_HPP

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <random>
#include <queue>
#include <cmath>
#include <iostream>
#include <memory>
#include <cstring>
#include <climits>
#include <chrono>
#include "hnswlib/hnswlib.h"

#define BTREE_M 3
#define BTREE_D 2

using namespace hnswlib;

class RangeHNSW {
public:
    // --- 兼容 main.cpp 的公开接口 ---
    int getMaxLayer() const { return maxLayer; }
    void addPoint(int key, int value, char* data) {}
    void erase(int key) {}
    void resize(size_t newMaxN) {}

    struct CompareByFirst {
        constexpr bool operator()(const std::pair<float, tableint>& a,
                                  const std::pair<float, tableint>& b) const noexcept {
            return a.first < b.first;
        }
    };

    // 大顶堆：维护当前找到的最佳 K 个结果
    typedef std::priority_queue<std::pair<float, tableint>, 
                                std::vector<std::pair<float, tableint>>, 
                                CompareByFirst> ResultHeap;

    // 小顶堆：维护待探索的候选点
    typedef std::priority_queue<std::pair<float, tableint>, 
                                std::vector<std::pair<float, tableint>>, 
                                std::greater<std::pair<float, tableint>>> CandidateHeap;

    RangeHNSW(int d, size_t eleNum, size_t maxEleNum, float* vecData, int* keyList, int* valueList, int m, int ef_con);
    ~RangeHNSW();

    std::priority_queue<std::pair<float, hnswlib::labeltype>> 
    queryRange(float *vecData, int rangeL, int rangeR, int k, int ef_s);

private:
    struct node {
        int entryPoint;
        int keynum;
        int key[BTREE_M];
        struct node* child[BTREE_M + 1];
        short int layer;
        node() : entryPoint(-1), keynum(0), layer(0) {
            memset(key, 0, sizeof(key));
            memset(child, 0, sizeof(child));
        }
    };

    struct Cluster {
        tableint representative;          // 聚类代表点
        std::vector<tableint> members;    // 聚类内成员点
        std::vector<tableint> neighbors;  // 邻居代表点
    };

    size_t eleCount;
    size_t maxNum;
    int dim;
    int M;
    int ef_construction;
    int maxLayer;

    node* root;
    float* vecDataRaw; 
    int* keyList_;
    int* valueList_;
    
    unsigned int *visited_array;
    unsigned int tag;
    
    hnswlib::L2Space space;
    DISTFUNC<float> fstdistfunc_;
    void *dist_func_param_;
    size_t data_size_;
    std::mt19937 eng;

    std::vector<Cluster> clusters;
    std::vector<int> pointToClusterIdx;

    // 内部函数
    inline float* getVectorPtr(tableint id) const { return vecDataRaw + (id * dim); }
    bool cmp(int a, int b);
    node* buildTree(int eleNum, const std::vector<int>& sortedArray);
    node* findHighNode(node* nd, int rangeL, int rangeR);
    void buildClusterGraph();
    void deleteTree(node* nd);
    
    // 核心搜索：支持全空间导航
    ResultHeap searchBestFirst(const float* query_vec, const std::vector<tableint>& eps, 
                               int rangeL, int rangeR, int ef);
};

// ==================== 实现部分 ====================

RangeHNSW::RangeHNSW(int d, size_t eleNum, size_t maxEleNum, float* vecData, 
                     int* keyList, int* valueList, int m, int ef_con)
    : eleCount(eleNum), maxNum(maxEleNum), dim(d), M(m), 
      ef_construction(ef_con), space(d), tag(0) {
    
    maxLayer = (int)floor(log((float)maxEleNum) / log(BTREE_D));
    data_size_ = space.get_data_size();
    fstdistfunc_ = space.get_dist_func();
    dist_func_param_ = space.get_dist_func_param();
    
    vecDataRaw = new float[maxEleNum * dim];
    keyList_ = new int[maxEleNum];
    valueList_ = new int[maxEleNum];
    visited_array = new unsigned int[maxEleNum]();
    pointToClusterIdx.assign(maxEleNum, -1);

    memcpy(vecDataRaw, vecData, eleNum * dim * sizeof(float));
    memcpy(keyList_, keyList, eleNum * sizeof(int));
    memcpy(valueList_, valueList, eleNum * sizeof(int));

    std::vector<int> sortedArray(eleNum);
    for(int i = 0; i < eleNum; i++) sortedArray[i] = i;
    std::sort(sortedArray.begin(), sortedArray.end(), [this](int a, int b){ return cmp(a, b); });

    root = buildTree(eleNum, sortedArray);
    buildClusterGraph();
}

RangeHNSW::~RangeHNSW() {
    delete[] vecDataRaw;
    delete[] keyList_;
    delete[] valueList_;
    delete[] visited_array;
    deleteTree(root);
}

void RangeHNSW::deleteTree(node* nd) {
    if (!nd) return;
    if (nd->layer > 0) {
        for (int i = 0; i <= nd->keynum; i++) deleteTree(nd->child[i]);
    }
    delete nd;
}

bool RangeHNSW::cmp(int a, int b) {
    if(valueList_[a] != valueList_[b]) return valueList_[a] < valueList_[b];
    return keyList_[a] < keyList_[b];
}

void RangeHNSW::buildClusterGraph() {
    // 适当的聚类大小，这里设为 logN 左右
    int targetSize = std::max(8, (int)log2(eleCount));
    for (size_t i = 0; i < eleCount; i += targetSize) {
        Cluster c;
        c.representative = i;
        clusters.push_back(std::move(c));
    }

    for (size_t i = 0; i < eleCount; i++) {
        int cIdx = i / targetSize;
        if (cIdx >= clusters.size()) cIdx = clusters.size() - 1;
        clusters[cIdx].members.push_back(i);
        pointToClusterIdx[i] = cIdx;
    }

    // 代表点连接：增加连接数 M 以保证连通性
    for (size_t i = 0; i < clusters.size(); i++) {
        CandidateHeap heap;
        float* v1 = getVectorPtr(clusters[i].representative);
        for (size_t j = 0; j < clusters.size(); j++) {
            if (i == j) continue;
            float d = fstdistfunc_(v1, getVectorPtr(clusters[j].representative), dist_func_param_);
            heap.push({d, (tableint)j});
        }
        // 使用 M*2 的连接强度增强代表点图的召回
        for (int k = 0; k < M * 2 && !heap.empty(); k++) {
            clusters[i].neighbors.push_back(clusters[heap.top().second].representative);
            heap.pop();
        }
    }
}

RangeHNSW::ResultHeap RangeHNSW::searchBestFirst(const float* query_vec, 
                                               const std::vector<tableint>& eps, 
                                               int rangeL, int rangeR, int ef) {
    tag++;
    ResultHeap top_candidates;      // 存储符合 Range 的 TopK
    CandidateHeap candidate_set;    // 用于图导航的小顶堆
    float lower_bound = std::numeric_limits<float>::max();

    for (tableint ep : eps) {
        if (ep < 0 || ep >= eleCount || visited_array[ep] == tag) continue;
        float dist = fstdistfunc_(query_vec, getVectorPtr(ep), dist_func_param_);
        visited_array[ep] = tag;
        
        // 所有点都参与导航
        candidate_set.push({dist, ep});
        
        // 只有符合范围的才进入结果集
        if (valueList_[ep] >= rangeL && valueList_[ep] <= rangeR) {
            top_candidates.push({dist, ep});
            if (top_candidates.size() > ef) top_candidates.pop();
            lower_bound = top_candidates.top().first;
        }
    }

    while (!candidate_set.empty()) {
        auto curr = candidate_set.top();
        candidate_set.pop();

        // 导航剪枝阈值：允许比当前最差结果稍远一点，增加搜索冗余度提高召回
        if (curr.first > lower_bound * 1.2 && top_candidates.size() >= ef) continue;

        int cIdx = pointToClusterIdx[curr.second];
        if (cIdx == -1) continue;

        // 1. 探索邻居代表点（导航的核心）
        for (tableint nb_rep : clusters[cIdx].neighbors) {
            if (visited_array[nb_rep] != tag) {
                visited_array[nb_rep] = tag;
                float d = fstdistfunc_(query_vec, getVectorPtr(nb_rep), dist_func_param_);
                
                if (top_candidates.size() < ef || d < lower_bound * 1.5) {
                    candidate_set.push({d, nb_rep});
                    if (valueList_[nb_rep] >= rangeL && valueList_[nb_rep] <= rangeR) {
                        top_candidates.push({d, nb_rep});
                        if (top_candidates.size() > ef) top_candidates.pop();
                        lower_bound = top_candidates.top().first;
                    }
                }
            }
        }
        
        // 2. 只有当代表点足够近时，才排查聚类内所有成员
        if (curr.first < lower_bound * 1.3 || top_candidates.size() < ef) {
            for (tableint m : clusters[cIdx].members) {
                if (visited_array[m] != tag) {
                    visited_array[m] = tag;
                    // 仅对符合范围的成员计算距离，减少 CPU 开销
                    if (valueList_[m] >= rangeL && valueList_[m] <= rangeR) {
                        float d = fstdistfunc_(query_vec, getVectorPtr(m), dist_func_param_);
                        if (top_candidates.size() < ef || d < lower_bound) {
                            top_candidates.push({d, m});
                            if (top_candidates.size() > ef) top_candidates.pop();
                            lower_bound = top_candidates.top().first;
                        }
                    }
                }
            }
        }
    }
    return top_candidates;
}

std::priority_queue<std::pair<float, hnswlib::labeltype>> 
RangeHNSW::queryRange(float *vecData, int rangeL, int rangeR, int k, int ef_s) {
    node* highNode = findHighNode(root, rangeL, rangeR);
    if (!highNode) return {};

    // 增加入口点覆盖范围：获取 B-Tree 子树下的一批入口点
    std::vector<tableint> eps;
    std::vector<node*> stack;
    stack.push_back(highNode);
    while(!stack.empty() && eps.size() < 64) {
        node* curr = stack.back();
        stack.pop_back();
        if (curr->entryPoint != -1) eps.push_back(curr->entryPoint);
        if (curr->layer > 0) {
            for(int i=0; i <= curr->keynum; i++) {
                if(curr->child[i]) stack.push_back(curr->child[i]);
            }
        }
    }

    ResultHeap rh = searchBestFirst(vecData, eps, rangeL, rangeR, std::max(k, ef_s));
    
    std::priority_queue<std::pair<float, hnswlib::labeltype>> result;
    while(!rh.empty()){
        result.push({rh.top().first, keyList_[rh.top().second]});
        rh.pop();
    }
    return result;
}

RangeHNSW::node* RangeHNSW::buildTree(int eleNum, const std::vector<int>& sortedArray) {
    if (eleNum == 0) return nullptr;
    std::queue<std::pair<std::pair<int,int>, node*>> q;
    for(int i = 0; i < eleNum; i++) {
        node *nd = new node();
        nd->entryPoint = sortedArray[i];
        nd->layer = 0;
        q.push({{i, i}, nd});
    }
    while(q.size() > 1) {
        size_t levelSize = q.size();
        while(levelSize > 0) {
            int numChild = std::min((int)levelSize, BTREE_M);
            node* parent = new node();
            parent->keynum = numChild - 1;
            for(int i=0; i<numChild; i++) {
                auto front = q.front(); q.pop();
                parent->child[i] = front.second;
                if(i > 0) parent->key[i-1] = sortedArray[front.first.first];
                if(i == 0) parent->entryPoint = front.second->entryPoint;
            }
            parent->layer = parent->child[0]->layer + 1;
            q.push({{0, 0}, parent});
            levelSize -= numChild;
        }
    }
    return q.front().second;
}

RangeHNSW::node* RangeHNSW::findHighNode(RangeHNSW::node* nd, int rangeL, int rangeR) {
    if (!nd || nd->layer == 0) return nd;
    int i = 0;
    while(i < nd->keynum && rangeL > valueList_[nd->key[i]]) i++;
    int j = 0;
    while(j < nd->keynum && rangeR > valueList_[nd->key[j]]) j++;
    if(i == j) return findHighNode(nd->child[i], rangeL, rangeR);
    return nd;
}

#endif