from my_collections import _OS, HeightWidth, ShapeItem
from lumotag_events import GameStatus, PlayerStatus
import numpy as np
import math

def get_targeted_player_details(analysis: dict[tuple, ShapeItem], gamestatus: GameStatus ) -> PlayerStatus | None:
    """
    Find which player is being targeted from detected tags and current gamestate.
    get the closest tag to the centre of the camera
    """
    if analysis and len(analysis) < 1:
        return None
    if gamestatus and len(gamestatus.players) < 1:
        return None
    # now lets find the tag closest to the middle for all image sizes
    shortest_distance = math.inf
    shortest_tag_shape: ShapeItem | None = None
    for resolution in analysis:
        img_centre_xy = [resolution[0] //2, resolution[1] //2]
        for tag_info in analysis[resolution]:
            distance_from_center = math.hypot(
                tag_info.centre_x_y[0] - img_centre_xy[0],
                tag_info.centre_x_y[1] - img_centre_xy[1]
            )
            if distance_from_center < shortest_distance:
                shortest_distance = distance_from_center
                shortest_tag_shape = tag_info

    if not shortest_tag_shape:
        return None
    # now we have to find what tag id matches with what player, via the gamestatus object on the server 
    for deviceid, player in gamestatus.players.items():
        if player.tag_id == str(shortest_tag_shape.decoded_id):
            return player
    # if we get here - we have a valid ID found but the player does not exist on the server, or the server not updated
    return None
            
            

def get_tag_health_mapping(gamestate: GameStatus) -> dict[str, int]:
    """
    Create a mapping of tag_id to health for quick lookups.
    
    Args:
        gamestate: Current game state
    
    Returns:
        Dictionary mapping tag_id (str) to health (int)
    """
    if not gamestate or not gamestate.players:
        return {}
    
    return {
        player.tag_id: player.health 
        for player in gamestate.players.values()
    }