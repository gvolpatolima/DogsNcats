<template>
  <div class="p-6 max-w-xs mx-auto bg-white rounded-xl shadow-lg flex items-center space-x-4">
    <img v-for="(image, index) in images" :key="index" :src="image" alt="Click to get a dog">
  </div>
</template>

<script>
import axios from 'axios';

export default {
  data() {
    return {
      maxImages: 3, // maximum number of images to display
      images: [], // array to store the image URLs
    };
  },
  created() {
    this.fetchDogImages(); // Fetch dog images on component creation
  },
  methods: {
    fetchDogImages() {
      // Fetch a random dog image URL from the API
      axios.get('https://dog.ceo/api/breeds/image/random')
        .then(response => {
          // Add the image URL to the array
          this.images.unshift(response.data.message);
          // Remove excess images if array length exceeds maxImages
          if (this.images.length > this.maxImages) {
            this.images.pop();
          }
        })
        .catch(error => {
          console.log('Failed to fetch dog image:', error);
        });
    },
  },
};
</script>
