# utils/network_builder.py
import networkx as nx
import pandas as pd
from models.reddit_scraper import RedditScraper
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse as sp
from networkx.algorithms import bipartite


def usercomment_tree(comments_df, include_root=True):
    """
    Create directed network of comments with optional root nodes.
    Add comments and replies of comments as nodes and connect them with edges.
    If include_root is True, add post nodes and connects posts with comments.

    See Week 4 Day 2 for more information on how to visualise usercomment_tree.
    """
    G = nx.DiGraph() # Directed graph
    
    # Add all comments as nodes
    for _, row in comments_df.iterrows(): # Iterate over rows
        G.add_node(row['comment_id'], # Add node with comment_id
                  author=row['author'], # Add author attribute
                  type='comment') # Add type attribute
        
        # Handle root comments
        if include_root and pd.isna(row['parent_id']): # if include root and parent_id is NaN
            G.add_node(row['post_id'], type='post') # Add post node
            G.add_edge(row['post_id'], row['comment_id']) # Add edge between post and comment
        else:
            parent = row['parent_id'] #if parent_id is not NaN, set parent to parent_id
            if parent in G: # if parent is in G
                G.add_edge(parent, row['comment_id']) # Add edge between parent and comment
    
    return G

def create_user_interaction_network(comments_df):
    """
    Create undirected network of user interactions. 
    Does not take into account the post author only builds network for comment and reply authors.
    """
    G = nx.Graph() # Undirected graph
    
    # Group comments by post to find interactions
    for post_id, post_comments in comments_df.groupby('post_id'): # group comments by posts
        # Create mapping of comments to authors
        comment_authors = post_comments.set_index('comment_id')['author'].to_dict() # get dictionary where key is comment_id and value is author
        
        # Find interactions through replies
        for _, comment in post_comments.iterrows(): # iterate over comments
            author = comment['author'] # get author
            parent_id = comment['parent_id'] # get parent_id
            
            # Skip deleted/None authors
            if author in ['[deleted]', None]: # skip comment if from deleted author or none
                continue
                
            # If parent exists, create edge between authors
            if parent_id in comment_authors: # if parent_id is in comment_authors
                parent_author = comment_authors[parent_id] # get parent author
                if parent_author not in ['[deleted]', None]: # if parent author is not deleted or none
                    G.add_edge(author, parent_author) # add edge between author and parent author
    
    return G

def create_user_post_network(comments_df):
    """
    Create bipartite network of users and posts.
    Connects authors of comments and replies to posts.
    Use bipartite_layout to visualize the network.
    """
    G = nx.Graph()
    
    # Add post nodes
    posts = comments_df['post_id'].unique() # get unique post_ids
    for post_id in posts: # iterate over posts
        G.add_node(post_id, bipartite=0) # Add post node
    
    # Add user nodes and edges
    for _, comment in comments_df.iterrows(): # iterate over comments
        author = comment['author'] # get author 
        if author not in ['[deleted]', None]: # if author is not deleted or none
            G.add_node(author, bipartite=1) # Add user node
            G.add_edge(author, comment['post_id']) # Add edge between user and post
    
    return G


def jaccard_similarity(x, y):
    """
    Calcuate Jaccard similarity between two sets.
    """
            
    intersection = len(set(x).intersection(set(y)))
    union = len(set(x).union(set(y)))
    return intersection / union

def find_similar_users(user_network, giant_component=True, top_n=None, metric='cosine'):
    """
    Find most similar users based on connections.
    - user_network: NetworkX graph of user interactions -> created with create_user_interaction_network
    - if giant_component is True, only consider the giant component of the network to find most similar users
    - if I use euclidean distance, I will get the user with the highest number of shared neighbors
    - if I use jaccard similarity, I will get the user with the highest proportion
    - if I use cosine similarity, I will get the user with the most similar connection patterns (connections are in similiar proportions) 


    """
    
    if giant_component: 
        # Get giant component
        giant = max(nx.connected_components(user_network), key=len)
        user_network = user_network.subgraph(giant).copy()
    
    # Get adjacency matrix
    adj_matrix = nx.to_numpy_array(user_network) # convert network to numpy array
    
    # Calculate cosine similarity - two users similiar if connected to the same set of users regardless of the number of connections
    if metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_similarity  
        user_similarities = cosine_similarity(adj_matrix) # calculate cosine similarity between users
    # Calculate jaccard similarity - two users similiar if share large proportion of their neighbours
    elif metric == 'jaccard':
        from sklearn.metrics.pairwise import pairwise_distances
        adj_matrix = adj_matrix.astype(bool).astype(int) # 1 if edge exists, 0 otherwise
        user_similarities = pairwise_distances(adj_matrix, metric=jaccard_similarity) # calculate jaccard similarity between sets of each users neighbors
    # Calculate euclidean distance - two users similiar if have high number of shared neigbours
    elif metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        user_similarities = euclidean_distances(adj_matrix)

    
    # Get user names
    users = list(user_network.nodes()) # get list of users
    
    # Find top similar pairs (excluding self-similarity)
    similar_pairs = [] # list to store similar pairs
    for i in range(len(users)): # iterate over users
        for j in range(i+1, len(users)): # iterate over users starting from i+1 to avoid calculating the same pair twice
            similar_pairs.append(( # append user pair and similarity score
                users[i], 
                users[j], 
                user_similarities[i,j]
            ))
    
    # Sort by similarity
    similar_pairs.sort(key=lambda x: x[2], reverse=True) # sort pairs by similarity score
    
    if top_n is None: # if top_n
        return similar_pairs # return all pairs
    else: # if top_n is not None
        return similar_pairs[:top_n] # return top_n pairs
    
def get_network_stats(G):
    """
    Calculate basic network metrics.
    - Nodes = total number of nodes in the network
    - Edges = total number of edges in the network
    - Density = density of the network = number of edges / number of possible edges
    - Components = number of connected components in the network
    
    """
    return {
        'nodes': len(G.nodes()),
        'edges': len(G.edges()),
        'density': nx.density(G),
        'components': nx.number_connected_components(G)
    }


def calculate_tree_width(tree):
    """
    Calculate width of a comment tree.
    - tree: NetworkX DiGraph representing the comment tree from usercomment_tree.
    """
    # Find the root node (node with no incoming edges)
    root = [node for node, degree in tree.in_degree() if degree == 0][0]
    
    # Get the shortest path lengths from the root (essentially the level of each node)
    levels = nx.single_source_shortest_path_length(tree, root)
    
    # Count the nodes at each level (i.e., the width at each level)
    level_counts = {}
    for level in levels.values():
        if level not in level_counts:
            level_counts[level] = 0
        level_counts[level] += 1
    
    # The width of the tree is the maximum number of nodes at any level
    return level_counts

def plot_user_interaction_network(comments_df:pd.DataFrame, n:int, title:str):
    """
    Function to plot the user interaction network and label the top n users with the highest degree centrality

    Inputs:
    - comments_df: pandas DataFrame containing the comments
    - n: number of top users to label
    """
    plt.figure()
    user_network = create_user_interaction_network(comments_df)  # undirected network
    pos_users = nx.spring_layout(user_network, k=1, iterations=200)

    degree_centrality = nx.degree_centrality(user_network)  # dictionary of degree centrality values for each node
    node_color = [degree_centrality[node] for node in user_network.nodes()]  # put the degree centrality values into a list in the order of the node ids
    node_size = [degree_centrality[node] * 10000 for node in user_network.nodes()]  # multiply the degree centrality values by 1000 to get the size of the nodes
    nx.draw(user_network, pos_users, node_size=node_size, node_color=node_color, cmap=plt.cm.plasma)  # colour nodes based on centrality because node_color=node_color

    # Label the top 3 nodes by in-degree centrality
    top_nodes = sorted(degree_centrality, key=degree_centrality.get, reverse=True)[:n]  # get node id of top 3 authors with highest degree centrality

    labels = {node: node for node in top_nodes}  # put names of top n authors into a dictionary (key and value is name of author)
    nx.draw_networkx_labels(user_network, pos_users, labels=labels, font_color='white', bbox=dict(facecolor='black', alpha=0.5), verticalalignment='bottom', horizontalalignment='left')
    plt.title(title)

def visualize_similar_users(user_network, title="User Network - Most Similar Users Highlighted", number_of_similar_users=5, metric='cosine'):
    """
    Visualize the most similar users in a user interaction network.

    Parameters:
    - user_network: networkx.classes.graph.Graph
    - number_of_similar_users: int, optional (default=10)
    The number of similar user pairs to visualize.
    - metric: str, optional (default='cosine')
    The metric used to find similar users. Options are 'cosine' or 'euclidean'.
    Cosine finds people that access network similarly
    Euclidean finds people that are connected with many other people
    """
    similar_users = find_similar_users(user_network, metric=metric)

    plt.figure(figsize=(10, 6))
    pos = nx.spring_layout(user_network)

    # Find the largest connected component
    giant_component = max(nx.connected_components(user_network), key=len)
    subgraph = user_network.subgraph(giant_component)

    # Draw full network in gray
    nx.draw(subgraph, pos, node_color='lightgray', edge_color='lightgray', width=1, node_size=100)

     # Highlight similar pairs
    for user1, user2, sim in similar_users[:number_of_similar_users]:
        # Draw edge in red with width proportional to similarity
        nx.draw_networkx_edges(subgraph, pos, 
                                edgelist=[(user1, user2)],
                                edge_color='red',
                                width=sim*3)
            
        # Label nodes
        nx.draw_networkx_labels(subgraph, pos, 
                                   labels={user1: user1, user2: user2},
                                   font_size=8)

    plt.title(f"{title} ({metric.capitalize()} Similarity)")
    plt.show()

    for user1, user2, sim in similar_users[:number_of_similar_users]:
        print(f"{user1} - {user2}: {sim:.3f}")

def comment_depth(df,posts,limit=100):
    """
    Calculate the depth of the comment tree for each post in the dataframe

    Inputs:
    - df: DataFrame containing comments
    - posts: list of dictionaries containing post information

    """
    depth_list = []
    for i in range(0,limit-1):
        post_comments = df[df['post_id'] == posts[i]['id']]
        comment_tree = usercomment_tree(post_comments, include_root=True)
        depth=depth = nx.dag_longest_path_length(comment_tree)
        depth_list.append(depth)
    return depth_list

def parts_of_biparte_network(matrix, user_label, post_label):
    """
    Plot bipartite network and its projections as their own networks:
        - Bipartite network
        - User-User projection: Eg. who is interacting with the same posts - users similarity based on post interaction
        - Post-Post projection Eg. how many users are interacting with the same posts - post similarity based on user interaction

    Inputs:
    - matrix: numpy array, the bipartite network matrix (rows are users, columns are posts - eg. 1 if user interacted with post, 0 otherwise)
    - user_label: list, the labels of the users
    - post_label: list, the labels of the posts

    ! May be useful to change the layout of the plots to better visualize the networks - use spring_layout to minimise energy of the network

    """
    user_user = np.dot(matrix, matrix.T) # User-User projection
    post_post = np.dot(matrix.T, matrix) # Post-Post projection

    A_sparse = sp.csr_matrix(matrix) # Convert to sparse matrix
    G_bipartite = bipartite.from_biadjacency_matrix(A_sparse) # Create bipartite graph

    user_map = {i: user_label[i] for i in range(len(user_label))} #  maps the indices of students (0, 1, 2) to their names
    post_map = {i+len(user_label): post_label[i] for i in range(len(post_label))} # maps the indices of courses to their names
    mapping = {**user_map, **post_map}

    G_bipartite = nx.relabel_nodes(G_bipartite, mapping) # Relabel nodes

    G_user = nx.from_numpy_array(user_user) # Create user-user graph
    G_user.remove_edges_from(nx.selfloop_edges(G_user)) # Remove self-loops (edge that connects a node to itself)

    G_posts = nx.from_numpy_array(post_post) # Create post-post graph
    G_posts.remove_edges_from(nx.selfloop_edges(G_posts)) # Remove self-loops (edge that connects a node to itself)

    # Add node labels after creation
    mapping = {i: user_label[i] for i in range(len(user_label))}
    G_user = nx.relabel_nodes(G_user, mapping)

    mapping = {i: post_label[i] for i in range(len(post_label))}
    G_posts = nx.relabel_nodes(G_posts, mapping)

    # Plotting
    plt.figure(figsize=(15, 5))

    ########################################################################################
    # Bipartite plot
    plt.subplot(131)
    pos = nx.spring_layout(G_bipartite)
    nx.draw_networkx_nodes(G_bipartite, pos,
                        nodelist=user_label, #[n for n in G_bipartite.nodes() if n in students],
                        node_color='lightblue',
                        node_size=300)

    nx.draw_networkx_nodes(G_bipartite, pos,
                        nodelist=post_label, #[n for n in G_bipartite.nodes() if n in courses],
                        node_color='lightgreen',
                        node_size=500)

    nx.draw_networkx_edges(G_bipartite, pos, alpha=0.5)
    nx.draw_networkx_labels(G_bipartite, pos, font_size=8)
    plt.title("Bipartite Network")

    ########################################################################################
    # Student projection
    plt.subplot(132)

    weights = [G_user[u][v]['weight'] for u,v in G_user.edges()] # visually reflect the relative importance or strength of the connections between nodes

    pos = nx.kamada_kawai_layout(G_user)

    nx.draw(G_user, pos,
            node_color='lightblue',
            node_size=300,
            width=[w/max(weights)*3 for w in weights],
            edge_color='gray',
            alpha=0.6)

    nx.draw_networkx_labels(G_user, pos, font_size=8)

    plt.title("User-User Network")

    ########################################################################################
    # Course projection
    plt.subplot(133)

    edge_labels = {(u, v): f"{G_posts[u][v]['weight']}" for u,v in G_posts.edges()}

    weights = [G_posts[u][v]['weight'] for u,v in G_posts.edges()] # visually reflect the relative importance or strength of the connections between nodes

    pos = nx.circular_layout(G_posts)

    nx.draw(G_posts, pos,
            node_color='lightgreen',
            node_size=500,
            width=[w/max(weights)*3 for w in weights], # width of edge depends on weight
            edge_color='gray',
            alpha=0.6)

    nx.draw_networkx_labels(G_posts, pos, font_size=10)
    nx.draw_networkx_edge_labels(G_posts, pos, edge_labels=edge_labels, font_size=8)

    plt.title("Post-Post Network")

    plt.tight_layout()
    plt.show()

    return None