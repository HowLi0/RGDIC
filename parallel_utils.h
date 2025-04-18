#ifndef PARALLEL_UTILS_H
#define PARALLEL_UTILS_H

#include <chrono>
#include <vector>
#include <string>
#include <iostream>
#include <functional>
#include <omp.h>

class ParallelPerformanceMonitor {
public:
    ParallelPerformanceMonitor(const std::string& name = "未命名任务") 
        : m_name(name), m_isRunning(false) {
        // 记录可用核心数
        m_availableCores = omp_get_max_threads();
    }
    
    void start() {
        m_isRunning = true;
        m_startTime = std::chrono::high_resolution_clock::now();
    }
    
    void stop() {
        if (!m_isRunning) return;
        
        auto endTime = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsedTime = endTime - m_startTime;
        m_executionTimes.push_back(elapsedTime.count());
        
        m_isRunning = false;
    }
    
    void reset() {
        m_executionTimes.clear();
        m_isRunning = false;
    }
    
    double getLastExecutionTime() const {
        return m_executionTimes.empty() ? 0.0 : m_executionTimes.back();
    }
    
    double getAverageExecutionTime() const {
        if (m_executionTimes.empty()) return 0.0;
        
        double sum = 0.0;
        for (double time : m_executionTimes) {
            sum += time;
        }
        return sum / m_executionTimes.size();
    }
    
    void runThreadScalingTest(std::function<void(int)> taskFunction, int maxThreads = 0) {
        if (maxThreads <= 0) {
            maxThreads = m_availableCores;
        }
        
        std::cout << "执行线程扩展测试 (" << m_name << "):" << std::endl;
        std::cout << "可用核心: " << m_availableCores << std::endl;
        
        std::vector<double> times;
        
        // 测试不同的线程数
        for (int numThreads = 1; numThreads <= maxThreads; numThreads++) {
            omp_set_num_threads(numThreads);
            
            // 运行预热
            taskFunction(numThreads);
            
            // 开始计时
            auto startTime = std::chrono::high_resolution_clock::now();
            
            // 执行任务
            taskFunction(numThreads);
            
            // 停止计时
            auto endTime = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> elapsedTime = endTime - startTime;
            
            times.push_back(elapsedTime.count());
            std::cout << "  线程: " << numThreads << " - 执行时间: " << elapsedTime.count() << " 秒" << std::endl;
        }
        
        // 计算加速比
        double baseTime = times[0];  // 单线程时间
        std::cout << "\n加速比 (相对于单线程):" << std::endl;
        
        for (int i = 0; i < times.size(); i++) {
            int numThreads = i + 1;
            double speedup = baseTime / times[i];
            double efficiency = speedup / numThreads * 100.0;
            
            std::cout << "  线程: " << numThreads 
                      << " - 加速比: " << speedup 
                      << " - 效率: " << efficiency << "%" << std::endl;
        }
    }
    
    void printReport() const {
        std::cout << "性能报告 (" << m_name << "):" << std::endl;
        std::cout << "  执行次数: " << m_executionTimes.size() << std::endl;
        std::cout << "  平均执行时间: " << getAverageExecutionTime() << " 秒" << std::endl;
        
        if (!m_executionTimes.empty()) {
            double minTime = *std::min_element(m_executionTimes.begin(), m_executionTimes.end());
            double maxTime = *std::max_element(m_executionTimes.begin(), m_executionTimes.end());
            
            std::cout << "  最小执行时间: " << minTime << " 秒" << std::endl;
            std::cout << "  最大执行时间: " << maxTime << " 秒" << std::endl;
        }
    }
    
private:
    std::string m_name;
    std::vector<double> m_executionTimes;
    std::chrono::time_point<std::chrono::high_resolution_clock> m_startTime;
    bool m_isRunning;
    int m_availableCores;
};

#endif // PARALLEL_UTILS_H