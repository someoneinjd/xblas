#pragma once

template <typename... Args>
void log(Args &&...args) {
#ifndef T2SP_NDEBUG
  ((std::cout << "[INFO] ") << ... << args) << "\n";
#endif
}
