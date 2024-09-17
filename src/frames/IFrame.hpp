/*
 *   Copyright (c) 2024 Jayme Queiroz CPGEI
 *   All rights reserved.
 *   IFrame is the base class for implementation of mmWave frame classes.
 */

#include <vector>
#include <cstdint>

class IFrame
{
public:
  virtual bool parse(const std::vector<uint8_t> &data) = 0;
  virtual void display() const = 0;
};
