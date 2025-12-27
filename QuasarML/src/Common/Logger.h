#pragma once

#include <spdlog/spdlog.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <memory>

namespace QuasarML {

class Logger {
public:
    static void init(const std::string& log_file = "quasar.log");
    static void shutdown();
    static std::shared_ptr<spdlog::logger>& get();
    
private:
    static std::shared_ptr<spdlog::logger> _logger;
};

#define LOG_TRACE(...) ::QuasarML::Logger::get()->trace(__VA_ARGS__)
#define LOG_DEBUG(...) ::QuasarML::Logger::get()->debug(__VA_ARGS__)
#define LOG_INFO(...)  ::QuasarML::Logger::get()->info(__VA_ARGS__)
#define LOG_WARN(...)  ::QuasarML::Logger::get()->warn(__VA_ARGS__)
#define LOG_ERROR(...) ::QuasarML::Logger::get()->error(__VA_ARGS__)
#define LOG_FATAL(...) ::QuasarML::Logger::get()->critical(__VA_ARGS__)

}
