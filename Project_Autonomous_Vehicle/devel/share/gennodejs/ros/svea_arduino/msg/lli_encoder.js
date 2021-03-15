// Auto-generated. Do not edit!

// (in-package svea_arduino.msg)


"use strict";

const _serializer = _ros_msg_utils.Serialize;
const _arraySerializer = _serializer.Array;
const _deserializer = _ros_msg_utils.Deserialize;
const _arrayDeserializer = _deserializer.Array;
const _finder = _ros_msg_utils.Find;
const _getByteLength = _ros_msg_utils.getByteLength;

//-----------------------------------------------------------

class lli_encoder {
  constructor(initObj={}) {
    if (initObj === null) {
      // initObj === null is a special case for deserialization where we don't initialize fields
      this.right_ticks = null;
      this.left_ticks = null;
      this.time_delta = null;
    }
    else {
      if (initObj.hasOwnProperty('right_ticks')) {
        this.right_ticks = initObj.right_ticks
      }
      else {
        this.right_ticks = 0;
      }
      if (initObj.hasOwnProperty('left_ticks')) {
        this.left_ticks = initObj.left_ticks
      }
      else {
        this.left_ticks = 0;
      }
      if (initObj.hasOwnProperty('time_delta')) {
        this.time_delta = initObj.time_delta
      }
      else {
        this.time_delta = 0;
      }
    }
  }

  static serialize(obj, buffer, bufferOffset) {
    // Serializes a message object of type lli_encoder
    // Serialize message field [right_ticks]
    bufferOffset = _serializer.uint8(obj.right_ticks, buffer, bufferOffset);
    // Serialize message field [left_ticks]
    bufferOffset = _serializer.uint8(obj.left_ticks, buffer, bufferOffset);
    // Serialize message field [time_delta]
    bufferOffset = _serializer.uint16(obj.time_delta, buffer, bufferOffset);
    return bufferOffset;
  }

  static deserialize(buffer, bufferOffset=[0]) {
    //deserializes a message object of type lli_encoder
    let len;
    let data = new lli_encoder(null);
    // Deserialize message field [right_ticks]
    data.right_ticks = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [left_ticks]
    data.left_ticks = _deserializer.uint8(buffer, bufferOffset);
    // Deserialize message field [time_delta]
    data.time_delta = _deserializer.uint16(buffer, bufferOffset);
    return data;
  }

  static getMessageSize(object) {
    return 4;
  }

  static datatype() {
    // Returns string type for a message object
    return 'svea_arduino/lli_encoder';
  }

  static md5sum() {
    //Returns md5sum for a message object
    return '41aecfbb5a17be4edf75b1acdf69d991';
  }

  static messageDefinition() {
    // Returns full string definition for message
    return `
    uint8 right_ticks
    uint8 left_ticks
    uint16 time_delta
    
    `;
  }

  static Resolve(msg) {
    // deep-construct a valid message object instance of whatever was passed in
    if (typeof msg !== 'object' || msg === null) {
      msg = {};
    }
    const resolved = new lli_encoder(null);
    if (msg.right_ticks !== undefined) {
      resolved.right_ticks = msg.right_ticks;
    }
    else {
      resolved.right_ticks = 0
    }

    if (msg.left_ticks !== undefined) {
      resolved.left_ticks = msg.left_ticks;
    }
    else {
      resolved.left_ticks = 0
    }

    if (msg.time_delta !== undefined) {
      resolved.time_delta = msg.time_delta;
    }
    else {
      resolved.time_delta = 0
    }

    return resolved;
    }
};

module.exports = lli_encoder;
