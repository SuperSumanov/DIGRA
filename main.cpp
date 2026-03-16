#include <iostream>
#include "hnswlib/hnswlib.h"
#include "DataMaker.hpp"
#include "TreeHNSW.hpp"

#include <chrono>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/resource.h>
#include <unistd.h>
#include <iomanip>

int stringTonum(char *ch){
    int len = strlen(ch);
    int res = 0;
    for(int i = 0; i< len; i++){
        res = res * 10 + ch[i] - '0';
    }
    return res;
}

// 获取当前内存使用（KB）
long getMemoryUsage() {
    struct rusage usage;
    getrusage(RUSAGE_SELF, &usage);
    return usage.ru_maxrss;  // 在Linux上返回KB
}

// 获取更详细的内存信息（从/proc/self/statm）
void printDetailedMemoryInfo(const std::string& label) {
    std::ifstream statm("/proc/self/statm");
    if (statm.is_open()) {
        long size, resident, share, text, lib, data, dt;
        statm >> size >> resident >> share >> text >> lib >> data >> dt;
        
        // 这些值都是以页为单位（通常4KB）
        long page_size = sysconf(_SC_PAGESIZE) / 1024;  // 转换为KB
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  [" << label << "] Memory Detail:" << std::endl;
        std::cout << "    Virtual: " << (size * page_size / 1024.0) << " MB" << std::endl;
        std::cout << "    Resident: " << (resident * page_size / 1024.0) << " MB" << std::endl;
        std::cout << "    Data: " << (data * page_size / 1024.0) << " MB" << std::endl;
        statm.close();
    }
}

// 打印内存使用情况
void printMemoryUsage(const std::string& stage) {
    long memory_kb = getMemoryUsage();
    double memory_mb = memory_kb / 1024.0;
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "=== Memory Usage [" << stage << "] ===" << std::endl;
    std::cout << "  RSS: " << memory_mb << " MB" << std::endl;
    
    // 获取系统内存信息
    std::ifstream meminfo("/proc/meminfo");
    if (meminfo.is_open()) {
        std::string line;
        long total_mem = 0, free_mem = 0, avail_mem = 0;
        
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                sscanf(line.c_str(), "MemTotal: %ld kB", &total_mem);
            } else if (line.find("MemFree:") == 0) {
                sscanf(line.c_str(), "MemFree: %ld kB", &free_mem);
            } else if (line.find("MemAvailable:") == 0) {
                sscanf(line.c_str(), "MemAvailable: %ld kB", &avail_mem);
            }
        }
        meminfo.close();
        
        std::cout << "  System Total: " << (total_mem / 1024.0) << " MB" << std::endl;
        std::cout << "  System Available: " << (avail_mem / 1024.0) << " MB" << std::endl;
        std::cout << "  Usage Ratio: " << (memory_mb / (total_mem / 1024.0) * 100) << "%" << std::endl;
    }
}

// 内存使用统计类
class MemoryTracker {
private:
    long peak_memory;
    long start_memory;
    std::vector<std::pair<std::string, long>> checkpoints;
    
public:
    MemoryTracker() : peak_memory(0), start_memory(0) {
        start_memory = getMemoryUsage();
        peak_memory = start_memory;
    }
    
    void checkpoint(const std::string& name) {
        long current = getMemoryUsage();
        checkpoints.push_back({name, current});
        if (current > peak_memory) peak_memory = current;
        
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Memory Checkpoint [" << name << "]: " 
                  << (current / 1024.0) << " MB"
                  << " (+" << ((current - start_memory) / 1024.0) << " MB from start)"
                  << std::endl;
    }
    
    void printSummary() {
        std::cout << "\n=== Memory Usage Summary ===" << std::endl;
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "  Start: " << (start_memory / 1024.0) << " MB" << std::endl;
        std::cout << "  Peak: " << (peak_memory / 1024.0) << " MB" << std::endl;
        std::cout << "  Peak Increase: " << ((peak_memory - start_memory) / 1024.0) << " MB" << std::endl;
        
        std::cout << "  Checkpoints:" << std::endl;
        long prev = start_memory;
        for (auto& cp : checkpoints) {
            std::cout << "    " << cp.first << ": " << (cp.second / 1024.0) << " MB"
                      << " (+" << ((cp.second - prev) / 1024.0) << " MB)" << std::endl;
            prev = cp.second;
        }
    }
    
    long getCurrentMemory() { return getMemoryUsage(); }
    long getPeakMemory() { return peak_memory; }
};

int main(int argc, char** argv){
    if (argc < 9) {
        std::cerr << "Usage: " << argv[0] 
                  << " baseNum queryNum dim k ef_con m baseFilename queryFilename" 
                  << std::endl;
        return 1;
    }
    
    puts("read begin");
    
    int baseNum = stringTonum(argv[1]);
    int queryNum = stringTonum(argv[2]);
    int dim = stringTonum(argv[3]);
    int k = stringTonum(argv[4]);
    int ef_con = stringTonum(argv[5]);
    int m = stringTonum(argv[6]);
    char *baseFilename = argv[7];
    char *queryFilename = argv[8];
    
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "\n=== Configuration ===" << std::endl;
    std::cout << "baseNum: " << baseNum << std::endl;
    std::cout << "queryNum: " << queryNum << std::endl;
    std::cout << "dim: " << dim << std::endl;
    std::cout << "k: " << k << std::endl;
    std::cout << "ef_con: " << ef_con << std::endl;
    std::cout << "m: " << m << std::endl;
    
    // 初始化内存追踪器
    MemoryTracker memoryTracker;
    printMemoryUsage("Program Start");
    memoryTracker.checkpoint("Before Data Loading");
    
    // DataMaker dataMaker(baseFilename, queryFilename, dataFilename, baseNum, queryNum, dim);
    DataMaker dataMaker(baseFilename, queryFilename, baseNum, queryNum, dim);
    
    memoryTracker.checkpoint("After Data Loading");
    printDetailedMemoryInfo("After Data Loading");
    
    auto start = std::chrono::high_resolution_clock::now();
    
    RangeHNSW rangeHnsw(dim, baseNum, baseNum, dataMaker.data, 
                        dataMaker.key, dataMaker.value, m, ef_con);
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    memoryTracker.checkpoint("After Index Building");
    printDetailedMemoryInfo("After Index Building");
    
    std::cout << "\nbuild time: " << elapsed.count() << " seconds" << std::endl;
    puts("buildOK");
    
    // 计算索引的理论内存占用
    size_t vector_memory = (size_t)baseNum * dim * sizeof(float);
    size_t key_memory = (size_t)baseNum * sizeof(int) * 2;  // keyList + valueList
    size_t graph_memory = (size_t)baseNum * (m * sizeof(tableint) + sizeof(linklistsizeint)) * (rangeHnsw.getMaxLayer() + 1);
    
    std::cout << "\n=== Theoretical Memory Breakdown ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "  Vectors: " << (vector_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Keys/Values: " << (key_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Graph: " << (graph_memory / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "  Total Theoretical: " 
              << ((vector_memory + key_memory + graph_memory) / (1024.0 * 1024.0)) 
              << " MB" << std::endl;
    
    std::vector<float> r = {0.01f, 0.05f, 0.1f, 0.2f, 0.4f};
    
    for(auto range : r){
        dataMaker.genRange(range, k);
        std::cout << "\n---------- " << std::fixed << std::setprecision(6) << range << " ----------" << std::endl;
        
        // 查询前内存检查
        memoryTracker.checkpoint("Before Query Range " + std::to_string(range));
        
        for(int ef = 25; ef <= 1000; ef += 25){
            float recall = 0.0f;
            double time = 0.0;
            
            auto query_start = std::chrono::high_resolution_clock::now();
            
            for(int i = 0; i < queryNum; i++){
                auto ans = dataMaker.getGt(i);
                
                auto start_q = std::chrono::high_resolution_clock::now();
                auto result = rangeHnsw.queryRange(
                    dataMaker.query + i * dim, 
                    dataMaker.qRange[i].first,
                    dataMaker.qRange[i].second, 
                    k, 
                    ef
                );
                auto end_q = std::chrono::high_resolution_clock::now();
                
                std::vector<int> r1, r2;
                while(!result.empty()){
                    r1.push_back(result.top().second);
                    result.pop();
                }
                for(int j = 0; j < k; j++) r2.push_back(ans[j].second);
                
                for(auto i1 : r1)
                    for(auto i2 : r2)
                        if(i1 == i2)
                            recall += 1.0f / k;
                
                std::chrono::duration<double> elapsed_q = end_q - start_q;
                time += elapsed_q.count();
            }
            
            auto query_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> query_elapsed = query_end - query_start;
            
            std::cout << std::fixed << std::setprecision(6);
            std::cout << "ef:" << ef << std::endl;
            std::cout << "recall:" << (recall / queryNum) << std::endl;
            std::cout << "time:" << time << std::endl;
            std::cout << "qps:" << (queryNum / time) << std::endl;
            
            // 每200个ef输出一次内存状态
            if (ef % 200 == 0) {
                memoryTracker.checkpoint("During Query ef=" + std::to_string(ef));
            }
        }
        
        // 每个range结束后输出内存状态
        memoryTracker.checkpoint("After Query Range " + std::to_string(range));
        printDetailedMemoryInfo("After Range " + std::to_string(range));
    }
    
    // 最终内存总结
    std::cout << "\n" << std::string(50, '=') << std::endl;
    memoryTracker.printSummary();
    printMemoryUsage("Program End");
    
    // 计算平均QPS和召回率统计（如果需要）
    std::cout << "\n=== Final Statistics ===" << std::endl;
    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Total queries processed: " << queryNum * r.size() * (1000/25) << std::endl;
    std::cout << "Peak memory usage: " << (memoryTracker.getPeakMemory() / 1024.0) << " MB" << std::endl;
    
    return 0;
}