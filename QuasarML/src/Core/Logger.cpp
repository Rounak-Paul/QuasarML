#include "Logger.h"
#include <iomanip>

namespace QuasarML {

// Static member definitions
std::unique_ptr<Logger> Logger::instance = nullptr;
std::mutex Logger::instanceMutex;

Logger::Logger() : currentLevel(LogLevel::TRACE), consoleOutput(true), fileOutput(false), lastQueryIndex(0) {
    // Initialize with default log file
    init_file_output("quasar.log");
}

Logger& Logger::get_instance() {
    std::lock_guard<std::mutex> lock(instanceMutex);
    if (!instance) {
        instance = std::unique_ptr<Logger>(new Logger());
    }
    return *instance;
}

Logger::~Logger() {
    if (logFile.is_open()) {
        logFile.close();
    }
}

std::string Logger::get_current_timestamp() const {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << "." << std::setfill('0') << std::setw(3) << ms.count();
    return ss.str();
}

std::string Logger::get_level_string(LogLevel level) const {
    switch (level) {
        case LogLevel::TRACE: return "TRACE";
        case LogLevel::DEBUG: return "DEBUG";
        case LogLevel::INFO:  return "INFO ";
        case LogLevel::WARN:  return "WARN ";
        case LogLevel::ERROR: return "ERROR";
        case LogLevel::FATAL: return "FATAL";
        default: return "UNKNOWN";
    }
}

std::string Logger::get_color_code(LogLevel level) const {
    switch (level) {
        case LogLevel::TRACE: return "\033[37m";  // White
        case LogLevel::DEBUG: return "\033[36m";  // Cyan
        case LogLevel::INFO:  return "\033[32m";  // Green
        case LogLevel::WARN:  return "\033[33m";  // Yellow
        case LogLevel::ERROR: return "\033[31m";  // Red
        case LogLevel::FATAL: return "\033[35m";  // Magenta
        default: return "\033[0m";               // Reset
    }
}

void Logger::write_log(LogLevel level, const std::string& message, const std::string& file, int line) {
    if (level < currentLevel) return;

    std::lock_guard<std::mutex> lock(logMutex);
    
    std::string timestamp = get_current_timestamp();
    std::string levelStr = get_level_string(level);
    std::string filename = file.substr(file.find_last_of("/\\") + 1);
    
    // Add to history for query system
    logHistory.emplace_back(timestamp, level, message, file, line);
    
    // Maintain max history size
    if (logHistory.size() > MAX_HISTORY_SIZE) {
        logHistory.pop_front();
        // Adjust lastQueryIndex if needed
        if (lastQueryIndex > 0) {
            lastQueryIndex--;
        }
    }
    
    std::stringstream logEntry;
    logEntry << "[" << timestamp << "] [" << levelStr << "] " 
             << filename << ":" << line << " - " << message;

    if (consoleOutput) {
        std::cout << get_color_code(level) << logEntry.str() << "\033[0m" << std::endl;
    }

    // Ensure file is open before writing
    if (fileOutput) {
        if (!logFile.is_open() && !logFilePath.empty()) {
            logFile.open(logFilePath, std::ios::app);
        }
        
        if (logFile.is_open()) {
            logFile << logEntry.str() << std::endl;
            logFile.flush();
        }
    }
}

std::vector<LogEntry> Logger::query_new_entries() {
    std::lock_guard<std::mutex> lock(logMutex);
    
    std::vector<LogEntry> new_entries;
    
    // Return all entries since last query
    for (size_t i = lastQueryIndex; i < logHistory.size(); ++i) {
        new_entries.push_back(logHistory[i]);
    }
    
    // Update query index
    lastQueryIndex = logHistory.size();
    
    return new_entries;
}

void Logger::set_log_level(LogLevel level) {
    std::lock_guard<std::mutex> lock(logMutex);
    currentLevel = level;
}

LogLevel Logger::get_log_level() const {
    return currentLevel;
}

void Logger::enable_console_output(bool enable) {
    std::lock_guard<std::mutex> lock(logMutex);
    consoleOutput = enable;
}

bool Logger::init_file_output(const std::string& filepath) {
    std::lock_guard<std::mutex> lock(logMutex);
    
    if (logFile.is_open()) {
        logFile.close();
    }
    
    logFile.open(filepath, std::ios::out | std::ios::trunc);
    if (logFile.is_open()) {
        fileOutput = true;
        logFilePath = filepath;
        return true;
    }
    
    fileOutput = false;
    return false;
}

void Logger::disable_file_output() {
    std::lock_guard<std::mutex> lock(logMutex);
    if (logFile.is_open()) {
        logFile.close();
    }
    fileOutput = false;
    logFilePath.clear();
}

void Logger::trace(const std::string& message, const std::string& file, int line) {
    write_log(LogLevel::TRACE, message, file, line);
}

void Logger::debug(const std::string& message, const std::string& file, int line) {
    write_log(LogLevel::DEBUG, message, file, line);
}

void Logger::info(const std::string& message, const std::string& file, int line) {
    write_log(LogLevel::INFO, message, file, line);
}

void Logger::warn(const std::string& message, const std::string& file, int line) {
    write_log(LogLevel::WARN, message, file, line);
}

void Logger::error(const std::string& message, const std::string& file, int line) {
    write_log(LogLevel::ERROR, message, file, line);
}

void Logger::fatal(const std::string& message, const std::string& file, int line) {
    write_log(LogLevel::FATAL, message, file, line);
}

} // namespace Quasar