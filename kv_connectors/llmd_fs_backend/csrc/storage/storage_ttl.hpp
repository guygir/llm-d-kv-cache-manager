// Copyright 2026 The llm-d Authors.
// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cerrno>
#include <array>
#include <mutex>
#include <chrono>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <system_error>
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

// Versioned namespace policy, shared with storage_ttl.py. No deadline renewal
// for v1; v2-idle renews retention. Stable stripe locks serialize writers and
// deleters.
class StorageTTLLock {
 public:
  explicit StorageTTLLock(const std::string& path, bool nonblocking = false) {
    if (path.find("/.llmd-ttl-") == std::string::npos) return;
    std::filesystem::path root;
    for (const auto& component : std::filesystem::path(path)) {
      root /= component;
      const auto name = component.string();
      const char* prefix = ".llmd-ttl-v1-";
      if (name.rfind(".llmd-ttl-v2-idle-", 0) == 0) {
        prefix = ".llmd-ttl-v2-idle-";
        idle_ = true;
      } else if (name.rfind(prefix, 0) != 0) {
        continue;
      }
      const auto suffix = name.substr(std::char_traits<char>::length(prefix));
      if (suffix.empty() ||
          suffix.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error("Invalid TTL namespace");
      seconds_ = std::stoll(suffix);
      if (seconds_ < 1 || seconds_ > 2147483647LL)
        throw std::runtime_error("Invalid TTL namespace duration");
      const auto bucket =
          std::filesystem::path(path).stem().string().substr(0, 3);
      if (bucket.size() != 3 ||
          bucket.find_first_not_of("0123456789abcdef") != std::string::npos)
        throw std::runtime_error("Invalid TTL block filename");
      std::filesystem::create_directories(root / ".locks");
      const auto lock = root / ".locks" / (bucket + ".lock");
      // POSIX record locks are process-owned. A process-local mutex also
      // serializes threads and prevents one FD close releasing a sibling lock.
      local_lock_ = std::unique_lock<std::mutex>(
          stripes()[std::stoul(bucket, nullptr, 16)],
          std::defer_lock);
      if (nonblocking) {
        if (!local_lock_.try_lock()) return;
      } else {
        local_lock_.lock();
      }
      fd_ = open(lock.c_str(), O_CREAT | O_RDWR | O_CLOEXEC, 0600);
      if (fd_ < 0)
        throw std::system_error(errno,
                                std::generic_category(),
                                "TTL lock open");
      struct flock request = {};
      request.l_type = F_WRLCK;
      request.l_whence = SEEK_SET;
      request.l_start = 0;
      request.l_len = 0;
      int result;
      do {
        result = fcntl(fd_, nonblocking ? F_SETLK : F_SETLKW, &request);
      } while (result < 0 && errno == EINTR);
      if (result < 0) {
        const int error = errno;
        close(fd_);
        fd_ = -1;
        if (nonblocking && (error == EACCES || error == EAGAIN)) return;
        throw std::system_error(error,
                                std::generic_category(),
                                "TTL publication lock");
      }
      acquired_ = true;
      return;
    }
  }
  bool acquired() const { return acquired_; }
  bool idle() const { return idle_; }
  StorageTTLLock(const StorageTTLLock&) = delete;
  StorageTTLLock& operator=(const StorageTTLLock&) = delete;
  ~StorageTTLLock() {
    if (fd_ >= 0) close(fd_);
  }
  bool expired(const std::string& path) const {
    if (seconds_ == 0) return false;
    struct stat info;
    if (stat(path.c_str(), &info) != 0) return true;
    const auto now = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         std::chrono::system_clock::now().time_since_epoch())
                         .count();
    const auto modified =
        static_cast<int64_t>(info.st_mtim.tv_sec) * 1000000000LL +
        info.st_mtim.tv_nsec;
    return now - modified >= seconds_ * 1000000000LL;
  }

 private:
  static std::array<std::mutex, 4096>& stripes() {
    static std::array<std::mutex, 4096> locks;
    return locks;
  }
  std::unique_lock<std::mutex> local_lock_;
  int fd_ = -1;
  int64_t seconds_ = 0;
  bool acquired_ = false;
  bool idle_ = false;
};

// Return an accepted descriptor, -1 for missing, or -2 for a busy bucket.
// Uses exactly the same local stripe and cross-process lock as native stores.
inline int try_open_idle_hit(const std::string& path) {
  StorageTTLLock guard(path, true);
  if (!guard.idle()) throw std::invalid_argument("Idle TTL namespace required");
  if (!guard.acquired()) return -2;
  int fd = open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
  if (fd < 0) {
    if (errno == ENOENT) return -1;
    throw std::system_error(errno, std::generic_category(), "TTL hit open");
  }
  struct stat info;
  struct timespec times[2] = {};
  times[0].tv_nsec = UTIME_NOW;
  times[1].tv_nsec = UTIME_OMIT;
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) ||
      futimens(fd, times) != 0) {
    const int error = errno ? errno : EINVAL;
    close(fd);
    throw std::system_error(error, std::generic_category(), "TTL hit renewal");
  }
  return fd;
}
