#include <gtest/gtest.h>
#include "my_tests.hpp"

TEST(MyLibraryTest, Addition) {
    ASSERT_EQ(add(2, 3), 5);
}
