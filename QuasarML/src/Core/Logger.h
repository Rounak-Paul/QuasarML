#pragma once
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <mutex>
#include <memory>
#include <chrono>
#include <vector>
#include <deque>

namespace QuasarML {

enum class LogLevel {
    TRACE = 0,
    DEBUG = 1,
    INFO = 2,
    WARN = 3,
    ERROR = 4,
    FATAL = 5
};

struct LogEntry {
    std::string timestamp;
    LogLevel level;
    std::string message;
    std::string file;
    int line;
    
    LogEntry(const std::string& ts, LogLevel lvl, const std::string& msg, 
             const std::string& f, int l)
        : timestamp(ts), level(lvl), message(msg), file(f), line(l) {}
};

class QS_API Logger {
private:
    static std::unique_ptr<Logger> instance;
    static std::mutex instanceMutex;
    
    std::ofstream logFile;
    std::mutex logMutex;
    LogLevel currentLevel;
    bool consoleOutput;
    bool fileOutput;
    std::string logFilePath;
    
    // History for query system
    std::deque<LogEntry> logHistory;
    size_t lastQueryIndex;
    static constexpr size_t MAX_HISTORY_SIZE = 5000;
    
    Logger();
    
    std::string get_current_timestamp() const;
    std::string get_level_string(LogLevel level) const;
    std::string get_color_code(LogLevel level) const;
    void write_log(LogLevel level, const std::string& message, const std::string& file, int line);
    
    // Helper for formatted logging
    template<typename... Args>
    std::string format_string(const std::string& format, Args&&... args) {
        std::string result = format;
        format_recursive(result, 0, std::forward<Args>(args)...);
        return result;
    }
    
    // Base case - no more arguments
    void format_recursive(std::string& result, size_t pos) {
        // Nothing to do - no more arguments
    }
    
    // Recursive case - process one argument at a time
    template<typename T, typename... Args>
    void format_recursive(std::string& result, size_t pos, T&& value, Args&&... args) {
        size_t placeholder_pos = result.find("{}", pos);
        if (placeholder_pos != std::string::npos) {
            // Convert value to string
            std::stringstream ss;
            ss << std::forward<T>(value);
            std::string value_str = ss.str();
            
            // Replace the placeholder
            result.replace(placeholder_pos, 2, value_str);
            
            // Continue with remaining arguments, adjusting position
            format_recursive(result, placeholder_pos + value_str.length(), std::forward<Args>(args)...);
        }
        // If no more placeholders found, ignore remaining arguments
    }

public:
    static Logger& get_instance();
    ~Logger();
    
    // Delete copy constructor and assignment operator
    Logger(const Logger&) = delete;
    Logger& operator=(const Logger&) = delete;
    
    void set_log_level(LogLevel level);
    LogLevel get_log_level() const;
    void enable_console_output(bool enable);
    bool init_file_output(const std::string& filepath);
    void disable_file_output();
    
    // Query interface for external systems (like ImGui logger)
    std::vector<LogEntry> query_new_entries();
    
    // Simple logging methods
    void trace(const std::string& message, const std::string& file = __FILE__, int line = __LINE__);
    void debug(const std::string& message, const std::string& file = __FILE__, int line = __LINE__);
    void info(const std::string& message, const std::string& file = __FILE__, int line = __LINE__);
    void warn(const std::string& message, const std::string& file = __FILE__, int line = __LINE__);
    void error(const std::string& message, const std::string& file = __FILE__, int line = __LINE__);
    void fatal(const std::string& message, const std::string& file = __FILE__, int line = __LINE__);
    
    // Template method for formatted logging
    template<typename... Args>
    void log(LogLevel level, const std::string& file, int line, const std::string& format, Args&&... args) {
        if (level < currentLevel) return;
        std::string formatted_message = format_string(format, std::forward<Args>(args)...);
        write_log(level, formatted_message, file, line);
    }
};

} // namespace QuasarML

// Conditional macros based on build mode
#ifndef QS_DEBUG
    // Release build - only warn, error, and fatal work
    #define LOG_TRACE(format, ...) ((void)0)
    #define LOG_DEBUG(format, ...) ((void)0)
    #define LOG_INFO(format, ...)  ((void)0)
    #define LOG_WARN(format, ...)  QuasarML::Logger::get_instance().log(QuasarML::LogLevel::WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)
    #define LOG_ERROR(format, ...) QuasarML::Logger::get_instance().log(QuasarML::LogLevel::ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)
    #define LOG_FATAL(format, ...) QuasarML::Logger::get_instance().log(QuasarML::LogLevel::FATAL, __FILE__, __LINE__, format, ##__VA_ARGS__)
#else
    // Debug build - all macros work with formatting
    #define LOG_TRACE(format, ...) QuasarML::Logger::get_instance().log(QuasarML::LogLevel::TRACE, __FILE__, __LINE__, format, ##__VA_ARGS__)
    #define LOG_DEBUG(format, ...) QuasarML::Logger::get_instance().log(QuasarML::LogLevel::DEBUG, __FILE__, __LINE__, format, ##__VA_ARGS__)
    #define LOG_INFO(format, ...)  QuasarML::Logger::get_instance().log(QuasarML::LogLevel::INFO, __FILE__, __LINE__, format, ##__VA_ARGS__)
    #define LOG_WARN(format, ...)  QuasarML::Logger::get_instance().log(QuasarML::LogLevel::WARN, __FILE__, __LINE__, format, ##__VA_ARGS__)
    #define LOG_ERROR(format, ...) QuasarML::Logger::get_instance().log(QuasarML::LogLevel::ERROR, __FILE__, __LINE__, format, ##__VA_ARGS__)
    #define LOG_FATAL(format, ...) QuasarML::Logger::get_instance().log(QuasarML::LogLevel::FATAL, __FILE__, __LINE__, format, ##__VA_ARGS__)
#endif