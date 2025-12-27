#include "Logger.h"

namespace QuasarML {

std::shared_ptr<spdlog::logger> Logger::_logger = nullptr;

void Logger::init(const std::string& log_file) {
    if (_logger) return;
    
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    console_sink->set_level(spdlog::level::warn);
    
    auto file_sink = std::make_shared<spdlog::sinks::basic_file_sink_mt>(log_file, true);
    file_sink->set_level(spdlog::level::trace);
    
    _logger = std::make_shared<spdlog::logger>("QuasarML", spdlog::sinks_init_list{console_sink, file_sink});
    _logger->set_level(spdlog::level::trace);
    _logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");
    
    spdlog::register_logger(_logger);
}

void Logger::shutdown() {
    if (_logger) {
        _logger->flush();
        spdlog::drop("QuasarML");
        _logger.reset();
    }
}

std::shared_ptr<spdlog::logger>& Logger::get() {
    if (!_logger) {
        init();
    }
    return _logger;
}

}
