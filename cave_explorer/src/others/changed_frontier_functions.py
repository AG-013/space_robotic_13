# NOT USING???

    
# def identify_frontiers(self): 
#     frontiers = []
#    # Extract map dimensions and data
#     width = self.current_map.info.width
#     height = self.current_map.info.height
    
#     map_array = np.array(self.current_map.data).reshape((height, width))

#     for i in range(height):
#         for j in range(width):
#             if map_array[i, j] == 0:  # Free space
#                 neighbors = self.get_neighbors(i, j, map_array)
#                 for n in neighbors:
#                     if map_array[n[0], n[1]] == -1:  # Unknown space
#                         frontiers.append((i, j))
#                         break
    
#     return frontiers


# def select_nearest_frontier(self, frontiers):
#     # MERGING FRONTIERS ##################################################################
#     merged_frontiers = []
#     threshold_distance = 50  # Distance threshold to consider frontiers as close

#     while frontiers:
#         # Select a random frontier
#         random_frontier = random.choice(frontiers)
#         frontiers.remove(random_frontier)

#         # Find close frontiers to the selected random frontier
#         close_frontiers = [random_frontier]
#         for frontier in frontiers[:]:
#             if compute_distance_between_points(random_frontier, frontier) < threshold_distance:
#                 close_frontiers.append(frontier)
#                 frontiers.remove(frontier)

#         # Merge the close frontiers into one
#         if close_frontiers:
#             avg_x = sum(f[0] for f in close_frontiers) / len(close_frontiers)
#             avg_y = sum(f[1] for f in close_frontiers) / len(close_frontiers)
#             merged_frontiers.append([avg_x, avg_y])
#     ##########################################################################################

#     # RETURN CLOSEST FRONTIER ################################################################
#     return merged_frontiers[0]