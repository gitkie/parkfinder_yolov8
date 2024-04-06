-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Generation Time: Mar 22, 2024 at 10:11 AM
-- Server version: 10.4.28-MariaDB
-- PHP Version: 8.0.28

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `db_acas`
--

-- --------------------------------------------------------

--
-- Table structure for table `carss`
--

CREATE TABLE `carss` (
  `id` int(11) NOT NULL,
  `area` varchar(255) NOT NULL,
  `x` int(11) DEFAULT NULL,
  `y` int(11) DEFAULT NULL,
  `date` date NOT NULL,
  `time` time NOT NULL,
  `status` enum('occupied','vacant') NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data for table `carss`
--

INSERT INTO `carss` (`id`, `area`, `x`, `y`, `date`, `time`, `status`) VALUES
(1, 'Area 1', 191, 252, '2024-03-22', '12:32:12', 'occupied'),
(2, 'Area 2', 323, 277, '2024-03-22', '12:32:12', 'occupied'),
(3, 'Area 3', NULL, NULL, '2024-03-22', '12:32:12', 'vacant'),
(4, 'Area 4', NULL, NULL, '2024-03-22', '12:32:12', 'vacant'),
(5, 'Area 5', 667, 272, '2024-03-22', '12:32:12', 'occupied');

--
-- Indexes for dumped tables
--

--
-- Indexes for table `carss`
--
ALTER TABLE `carss`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `area` (`area`);

--
-- AUTO_INCREMENT for dumped tables
--

--
-- AUTO_INCREMENT for table `carss`
--
ALTER TABLE `carss`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=814;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
